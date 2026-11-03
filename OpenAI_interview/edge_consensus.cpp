// 分布式边缘协同 Agent 运行时 —— LWW-CRDT + 动态 Quorum + 降级状态机 参考实现
//
// 编译: g++ -std=c++17 -O2 edge_consensus.cpp -o edge_consensus
// 运行: ./edge_consensus
//
// 三个可运行系统件:
//   1. LWWRegister / LWWMap : LWW-Element CRDT, 带 Lamport 时间戳,
//                             合并满足交换律/结合律/幂等 -> 最终一致。
//   2. dynamic_quorum_mode   : 少数派分区自动进只读 Safe Mode, 防脑裂。
//   3. degrade_mode          : RTT -> Agent 降级状态机 (Collaborative/Local/Offline)。

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>

// ─────────────────────────────────────────────────────────────
// Lamport 时间戳: (counter, node_id) 全序. counter 大者胜; 平局用 node_id 决胜。
// ─────────────────────────────────────────────────────────────
struct LamportTS {
    uint64_t counter = 0;
    uint32_t node_id = 0; 
    // 严格全序比较, 保证任意两节点对 "谁更新" 的判定一致 (LWW 无歧义)
    bool newer_than(const LamportTS& o) const {
        if (counter != o.counter) return counter > o.counter;
        return node_id > o.node_id;
    }
};

// ─────────────────────────────────────────────────────────────
// 1. LWW-Register: 单个值的 Last-Write-Wins CRDT
// ─────────────────────────────────────────────────────────────
template <typename V>
struct LWWRegister {
    V value{};
    LamportTS ts{};

    // 合并: 保留时间戳更新的一方。交换律 + 结合律 + 幂等 => 收敛。
    void merge(const LWWRegister& other) {
        if (other.ts.newer_than(ts)) {
            value = other.value;
            ts = other.ts;
        }
    }
};

// LWW-Map: key -> LWWRegister, 松耦合协同状态 (如各节点上报的坐标/意图)
template <typename V>
class LWWMap {
public:
    void set(const std::string& key, const V& v, LamportTS ts) {
        auto& reg = data_[key];
        LWWRegister<V> incoming{v, ts};
        reg.merge(incoming);
    }

    void merge(const LWWMap& other) {
        for (const auto& [k, reg] : other.data_) {
            data_[k].merge(reg);
        }
    }

    bool get(const std::string& key, V& out) const {
        auto it = data_.find(key);
        if (it == data_.end()) return false;
        out = it->second.value;
        return true;
    }

    void dump(const char* name) const {
        std::cout << "  [" << name << "] ";
        for (const auto& [k, reg] : data_) {
            std::cout << k << "=" << reg.value << "(ts=" << reg.ts.counter
                      << "," << reg.ts.node_id << ") ";
        }
        std::cout << "\n";
    }

private:
    std::map<std::string, LWWRegister<V>> data_;
};

// ─────────────────────────────────────────────────────────────
// 2. 动态 Quorum: 分区节点数是否构成多数派, 否则进只读 Safe Mode 防脑裂
// ─────────────────────────────────────────────────────────────
bool has_quorum(int reachable_nodes, int total_nodes) {
    int majority = total_nodes / 2 + 1;  // ⌈N/2⌉ 的等价整数写法
    return reachable_nodes >= majority;
}

// ─────────────────────────────────────────────────────────────
// 3. 降级状态机: 网络 RTT -> Agent 行为模式
// ─────────────────────────────────────────────────────────────
enum class AgentMode { Collaborative, LocalGroup, AutonomousSafe };

const char* to_string(AgentMode m) {
    switch (m) {
        case AgentMode::Collaborative:  return "Collaborative (全功能协同)";
        case AgentMode::LocalGroup:     return "LocalGroup (局部共识)";
        case AgentMode::AutonomousSafe: return "AutonomousSafe (离线安全自治)";
    }
    return "?";
}

AgentMode degrade_mode(double rtt_ms, bool connected) {
    if (!connected)          return AgentMode::AutonomousSafe;
    if (rtt_ms < 100.0)      return AgentMode::Collaborative;
    if (rtt_ms <= 2000.0)    return AgentMode::Collaborative;  // 尚可全功能
    return AgentMode::LocalGroup;                              // RTT > 2000ms
}

int main() {
    // ---- 1) CRDT 收敛演示: A、B 断连各自更新, 重连合并 -> 最终一致 ----
    std::cout << "== LWW-CRDT convergence ==\n";
    LWWMap<int> nodeA, nodeB;

    // 离线期间各自本地决策 (counter, node_id): A=1, B=2
    nodeA.set("robot_x", 10, {5, 1});   // A 在逻辑时刻 5 写 x=10
    nodeB.set("robot_x", 20, {7, 2});   // B 在逻辑时刻 7 写 x=20 (更新)
    nodeA.set("intent", 1, {9, 1});     // 只有 A 有的 key
    nodeB.set("battery", 88, {3, 2});   // 只有 B 有的 key
    nodeA.dump("A before");
    nodeB.dump("B before");

    // 重连: 双向合并。验证顺序无关 (交换律): A<-B 与 B<-A 结果相同
    LWWMap<int> a2 = nodeA, b2 = nodeB;
    a2.merge(nodeB);
    b2.merge(nodeA);
    a2.dump("A after ");
    b2.dump("B after ");

    int ax, bx;
    a2.get("robot_x", ax);
    b2.get("robot_x", bx);
    std::cout << "  converged robot_x: A=" << ax << " B=" << bx
              << (ax == bx ? "  [OK 一致, 取更新的 B=20]" : "  [MISMATCH]") << "\n";

    // ---- 2) 动态 Quorum 脑裂防护 ----
    std::cout << "\n== Dynamic Quorum (total=5) ==\n";
    for (int reachable : {5, 3, 2, 1}) {
        bool q = has_quorum(reachable, 5);
        std::cout << "  reachable=" << reachable << " -> "
                  << (q ? "QUORUM: 可写, 继续决策"
                        : "NO QUORUM: 进入只读 Safe Mode 防脑裂")
                  << "\n";
    }

    // ---- 3) 降级状态机 ----
    std::cout << "\n== Graceful Degradation ==\n";
    struct Net { double rtt; bool connected; };
    for (Net n : {Net{50, true}, Net{2500, true}, Net{0, false}}) {
        std::cout << "  rtt=" << n.rtt << "ms connected=" << (n.connected ? "yes" : "no")
                  << " -> " << to_string(degrade_mode(n.rtt, n.connected)) << "\n";
    }
    return 0;
}
