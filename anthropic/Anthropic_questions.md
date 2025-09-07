# Anthropic MLE Interview Questions - Detailed Solutions

## Problem 1: Constitutional AI Training Implementation

### Question
"How would you implement constitutional AI training for a language model?"

### Concept Overview
Constitutional AI (CAI) is Anthropic's approach to training AI systems that are helpful, harmless, and honest. The key concepts include:

- **Self-supervision**: The model learns to self-correct and improve
- **Constitutional principles**: Defining basic rules for AI behavior
- **Harmlessness vs helpfulness**: Balancing safety with utility
- **RLHF integration**: Combining human feedback with reinforcement learning

### Implementation Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import numpy as np

class ConstitutionalAITrainer:
    def __init__(self, base_model_name: str, constitutional_principles: List[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.constitutional_principles = constitutional_principles
        
        # Constitutional principles encoding
        self.principle_embeddings = self._encode_principles()
        
    def _encode_principles(self) -> torch.Tensor:
        """Encode constitutional principles into embeddings"""
        principle_tokens = []
        for principle in self.constitutional_principles:
            tokens = self.tokenizer(principle, return_tensors='pt', padding=True, truncation=True)
            principle_tokens.append(tokens['input_ids'])
        
        # Create principle embeddings matrix
        max_len = max(tokens.shape[1] for tokens in principle_tokens)
        principle_matrix = torch.zeros(len(principle_tokens), max_len)
        
        for i, tokens in enumerate(principle_tokens):
            principle_matrix[i, :tokens.shape[1]] = tokens.squeeze()
            
        return principle_matrix

class ConstitutionalTrainingPipeline:
    def __init__(self, model, tokenizer, principles):
        self.model = model
        self.tokenizer = tokenizer
        self.principles = principles
        
    def supervised_fine_tuning(self, dataset):
        """Phase 1: Supervised fine-tuning on helpful responses"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)
        
        for batch in dataset:
            inputs = self.tokenizer(batch['prompts'], return_tensors='pt', padding=True)
            targets = self.tokenizer(batch['responses'], return_tensors='pt', padding=True)
            
            outputs = self.model(**inputs, labels=targets['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    def constitutional_training(self, dataset):
        """Phase 2: Constitutional AI training with critique and revision"""
        
        for batch in dataset:
            # Step 1: Generate initial responses
            initial_responses = self._generate_responses(batch['prompts'])
            
            # Step 2: Generate critiques based on constitutional principles
            critiques = self._generate_critiques(batch['prompts'], initial_responses)
            
            # Step 3: Generate revised responses
            revised_responses = self._generate_revisions(
                batch['prompts'], initial_responses, critiques
            )
            
            # Step 4: Train on revised responses
            self._train_on_revisions(batch['prompts'], revised_responses)
    
    def _generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate initial responses to prompts"""
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response[len(prompt):].strip())
        
        return responses
    
    def _generate_critiques(self, prompts: List[str], responses: List[str]) -> List[str]:
        """Generate critiques using constitutional principles"""
        critiques = []
        
        for prompt, response in zip(prompts, responses):
            critique_prompt = f"""
            Please critique the following response according to these constitutional principles:
            {chr(10).join(f"- {p}" for p in self.principles)}
            
            Original prompt: {prompt}
            Response: {response}
            
            Critique:"""
            
            inputs = self.tokenizer(critique_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    temperature=0.3,
                    do_sample=True
                )
            
            critique = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            critiques.append(critique[len(critique_prompt):].strip())
        
        return critiques
    
    def _generate_revisions(self, prompts: List[str], responses: List[str], 
                          critiques: List[str]) -> List[str]:
        """Generate revised responses based on critiques"""
        revisions = []
        
        for prompt, response, critique in zip(prompts, responses, critiques):
            revision_prompt = f"""
            Original prompt: {prompt}
            Previous response: {response}
            Critique: {critique}
            
            Please provide a revised response that addresses the critique:"""
            
            inputs = self.tokenizer(revision_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.5,
                    do_sample=True
                )
            
            revision = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            revisions.append(revision[len(revision_prompt):].strip())
        
        return revisions
    
    def _train_on_revisions(self, prompts: List[str], revisions: List[str]):
        """Train the model on revised responses"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        
        for prompt, revision in zip(prompts, revisions):
            full_text = f"{prompt} {revision}"
            inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True)
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# Example usage
constitutional_principles = [
    "Be helpful and provide accurate information",
    "Avoid harmful, illegal, or unethical content",
    "Respect human autonomy and dignity",
    "Be honest about limitations and uncertainty",
    "Protect privacy and confidentiality"
]

# Initialize trainer
trainer = ConstitutionalAITrainer("gpt2", constitutional_principles)
pipeline = ConstitutionalTrainingPipeline(trainer.model, trainer.tokenizer, constitutional_principles)

# Training process would involve:
# 1. Supervised fine-tuning on helpful examples
# 2. Constitutional training with critique and revision
# 3. RLHF integration for preference learning
```

### Evaluation Metrics

```python
class ConstitutionalEvaluator:
    def __init__(self, principles):
        self.principles = principles
    
    def evaluate_harmlessness(self, responses: List[str]) -> float:
        """Evaluate how well responses adhere to safety principles"""
        harmlessness_scores = []
        
        for response in responses:
            # Check for harmful content patterns
            harmful_patterns = [
                "violence", "illegal", "discriminatory", "misleading"
            ]
            
            harm_score = 0
            for pattern in harmful_patterns:
                if pattern.lower() in response.lower():
                    harm_score += 1
            
            # Higher score = more harmless
            harmlessness_scores.append(1 - (harm_score / len(harmful_patterns)))
        
        return np.mean(harmlessness_scores)
    
    def evaluate_helpfulness(self, prompts: List[str], responses: List[str]) -> float:
        """Evaluate how helpful responses are"""
        # This would typically involve human evaluation or trained classifiers
        # For demonstration, we'll use response length and relevance heuristics
        helpfulness_scores = []
        
        for prompt, response in zip(prompts, responses):
            # Basic heuristics (in practice, use more sophisticated methods)
            length_score = min(len(response) / 100, 1.0)  # Normalize by expected length
            relevance_score = len(set(prompt.split()) & set(response.split())) / len(prompt.split())
            
            helpfulness_scores.append((length_score + relevance_score) / 2)
        
        return np.mean(helpfulness_scores)
    
    def consistency_metric(self, responses: List[str]) -> float:
        """Evaluate consistency in following constitutional principles"""
        # Measure how consistently the model applies principles
        consistency_scores = []
        
        for response in responses:
            principle_adherence = []
            for principle in self.principles:
                # Simplified adherence check
                adherence = 1.0 if any(
                    keyword in response.lower() 
                    for keyword in principle.lower().split()
                ) else 0.0
                principle_adherence.append(adherence)
            
            consistency_scores.append(np.mean(principle_adherence))
        
        return np.mean(consistency_scores)
```

---

## Problem 2: Real-time Harmful Output Detection System

### Question
"Design a system to detect and mitigate harmful outputs in real-time."

### System Architecture

```python
import asyncio
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class HarmType(Enum):
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    ILLEGAL_ACTIVITY = "illegal_activity"
    SEXUAL_CONTENT = "sexual_content"

@dataclass
class DetectionResult:
    harm_type: Optional[HarmType]
    confidence: float
    explanation: str
    should_block: bool
    suggested_revision: Optional[str] = None

class MultiLayerHarmDetector:
    def __init__(self):
        # Layer 1: Keyword filtering
        self.keyword_filters = self._load_keyword_filters()
        
        # Layer 2: Semantic analysis models
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Layer 3: Context understanding
        self.context_analyzer = AutoModelForSequenceClassification.from_pretrained(
            "unitary/toxic-bert"
        )
        self.context_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        
        # Adaptive thresholds
        self.thresholds = {
            HarmType.VIOLENCE: 0.7,
            HarmType.HATE_SPEECH: 0.8,
            HarmType.MISINFORMATION: 0.6,
            HarmType.PRIVACY_VIOLATION: 0.9,
            HarmType.ILLEGAL_ACTIVITY: 0.85,
            HarmType.SEXUAL_CONTENT: 0.75
        }
        
    def _load_keyword_filters(self) -> Dict[HarmType, List[str]]:
        """Load keyword filters for different harm types"""
        return {
            HarmType.VIOLENCE: [
                "kill", "murder", "assault", "weapon", "bomb", "violence",
                "hurt", "harm", "attack", "fight", "destroy"
            ],
            HarmType.HATE_SPEECH: [
                "hate", "discriminate", "racist", "sexist", "bigot",
                "inferior", "worthless", "scum"
            ],
            HarmType.PRIVACY_VIOLATION: [
                "social security", "credit card", "password", "address",
                "phone number", "personal information"
            ],
            HarmType.ILLEGAL_ACTIVITY: [
                "drug dealing", "money laundering", "fraud", "theft",
                "hacking", "piracy", "smuggling"
            ],
            HarmType.SEXUAL_CONTENT: [
                "explicit", "pornographic", "sexual", "nude", "intimate"
            ]
        }
    
    async def detect_harm(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """Main detection pipeline with multiple layers"""
        
        # Layer 1: Keyword filtering (fast)
        keyword_result = self._keyword_screening(text)
        if keyword_result.should_block:
            return keyword_result
        
        # Layer 2: Semantic analysis (medium speed)
        semantic_result = await self._semantic_analysis(text)
        if semantic_result.should_block:
            return semantic_result
        
        # Layer 3: Context understanding (slower but more accurate)
        context_result = await self._context_analysis(text, context)
        
        return context_result
    
    def _keyword_screening(self, text: str) -> DetectionResult:
        """Fast keyword-based filtering"""
        text_lower = text.lower()
        max_confidence = 0.0
        detected_harm = None
        
        for harm_type, keywords in self.keyword_filters.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = min(matches / 3.0, 1.0)  # Normalize by expected keyword count
            
            if confidence > max_confidence:
                max_confidence = confidence
                detected_harm = harm_type
        
        should_block = max_confidence > 0.8  # High threshold for keyword-only blocking
        
        return DetectionResult(
            harm_type=detected_harm,
            confidence=max_confidence,
            explanation=f"Keyword screening detected potential {detected_harm.value if detected_harm else 'none'}" 
                       f" with confidence {max_confidence:.2f}",
            should_block=should_block
        )
    
    async def _semantic_analysis(self, text: str) -> DetectionResult:
        """ML-based semantic harm detection"""
        try:
            # Use toxicity classifier
            result = self.toxicity_classifier(text)
            
            # Map toxicity labels to harm types
            label_mapping = {
                'TOXIC': HarmType.HATE_SPEECH,
                'SEVERE_TOXIC': HarmType.VIOLENCE,
                'THREAT': HarmType.VIOLENCE,
                'INSULT': HarmType.HATE_SPEECH,
                'IDENTITY_HATE': HarmType.HATE_SPEECH
            }
            
            if result[0]['label'] in label_mapping:
                harm_type = label_mapping[result[0]['label']]
                confidence = result[0]['score']
                should_block = confidence > self.thresholds.get(harm_type, 0.7)
                
                return DetectionResult(
                    harm_type=harm_type,
                    confidence=confidence,
                    explanation=f"Semantic analysis detected {harm_type.value} "
                               f"with confidence {confidence:.2f}",
                    should_block=should_block
                )
            
        except Exception as e:
            print(f"Semantic analysis error: {e}")
        
        return DetectionResult(
            harm_type=None,
            confidence=0.0,
            explanation="Semantic analysis found no harmful content",
            should_block=False
        )
    
    async def _context_analysis(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """Deep context understanding for nuanced detection"""
        
        # Combine text with context for better understanding
        full_text = f"{context} {text}" if context else text
        
        # Tokenize and analyze
        inputs = self.context_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.context_analyzer(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            
            # Assume binary classification (harmful/not harmful)
            harm_probability = probabilities[0][1].item()
        
        # Determine harm type based on content analysis
        harm_type = self._classify_harm_type(text, harm_probability)
        should_block = harm_probability > self.thresholds.get(harm_type, 0.7) if harm_type else False
        
        return DetectionResult(
            harm_type=harm_type,
            confidence=harm_probability,
            explanation=f"Context analysis determined harm probability: {harm_probability:.2f}",
            should_block=should_block,
            suggested_revision=self._suggest_revision(text, harm_type) if should_block else None
        )
    
    def _classify_harm_type(self, text: str, confidence: float) -> Optional[HarmType]:
        """Classify the specific type of harm based on text content"""
        if confidence < 0.5:
            return None
        
        text_lower = text.lower()
        
        # Simple classification based on content patterns
        if any(word in text_lower for word in ["violence", "kill", "hurt", "weapon"]):
            return HarmType.VIOLENCE
        elif any(word in text_lower for word in ["hate", "discriminate", "racist"]):
            return HarmType.HATE_SPEECH
        elif any(word in text_lower for word in ["false", "misinformation", "lie"]):
            return HarmType.MISINFORMATION
        elif any(word in text_lower for word in ["personal", "private", "confidential"]):
            return HarmType.PRIVACY_VIOLATION
        elif any(word in text_lower for word in ["illegal", "criminal", "unlawful"]):
            return HarmType.ILLEGAL_ACTIVITY
        else:
            return HarmType.VIOLENCE  # Default fallback
    
    def _suggest_revision(self, text: str, harm_type: Optional[HarmType]) -> str:
        """Suggest a safer revision of harmful content"""
        if not harm_type:
            return text
        
        revision_templates = {
            HarmType.VIOLENCE: "I can't provide information about violence. Instead, let me help with...",
            HarmType.HATE_SPEECH: "I don't engage with discriminatory content. I'd be happy to discuss...",
            HarmType.MISINFORMATION: "I should clarify that this information may be inaccurate. Let me provide...",
            HarmType.PRIVACY_VIOLATION: "I can't share personal information. I can help with general information about...",
            HarmType.ILLEGAL_ACTIVITY: "I can't provide guidance on illegal activities. I can suggest legal alternatives...",
            HarmType.SEXUAL_CONTENT: "I can't provide explicit content. I can help with other topics..."
        }
        
        return revision_templates.get(harm_type, "I can't provide that information. Let me help with something else.")

class RealTimeHarmMitigation:
    def __init__(self):
        self.detector = MultiLayerHarmDetector()
        self.processing_queue = asyncio.Queue()
        self.confidence_threshold = 0.7
        
    async def process_output(self, text: str, context: Optional[str] = None) -> Tuple[str, bool]:
        """Process output in real-time with low latency"""
        
        # Quick pre-screening
        if len(text) < 10:  # Very short text unlikely to be harmful
            return text, True
        
        # Detect harm
        detection_result = await self.detector.detect_harm(text, context)
        
        if detection_result.should_block:
            # Return revised content or block
            revised_text = detection_result.suggested_revision or "I can't provide that information."
            return revised_text, False
        else:
            return text, True
    
    def adapt_thresholds(self, feedback: Dict[HarmType, List[Tuple[float, bool]]]):
        """Adapt detection thresholds based on user feedback"""
        for harm_type, feedback_data in feedback.items():
            # Calculate optimal threshold based on feedback
            # feedback_data: List of (confidence, was_actually_harmful) tuples
            
            true_positives = sum(1 for conf, actual in feedback_data if actual and conf > self.detector.thresholds[harm_type])
            false_positives = sum(1 for conf, actual in feedback_data if not actual and conf > self.detector.thresholds[harm_type])
            
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                
                # Adjust threshold based on precision
                if precision < 0.8:  # Too many false positives
                    self.detector.thresholds[harm_type] = min(0.95, self.detector.thresholds[harm_type] + 0.05)
                elif precision > 0.95:  # Very high precision, can lower threshold
                    self.detector.thresholds[harm_type] = max(0.5, self.detector.thresholds[harm_type] - 0.05)

# Example usage
async def main():
    mitigator = RealTimeHarmMitigation()
    
    test_texts = [
        "How can I help you today?",
        "I hate all people from that country",
        "Here's how to make a bomb",
        "The weather is nice today"
    ]
    
    for text in test_texts:
        result, is_safe = await mitigator.process_output(text)
        print(f"Original: {text}")
        print(f"Result: {result}")
        print(f"Safe: {is_safe}")
        print("---")

# Run example
# asyncio.run(main())
```

### Performance Optimization

```python
class OptimizedHarmDetector:
    def __init__(self):
        # Use quantized models for faster inference
        self.fast_classifier = self._load_quantized_model()
        
        # Implement caching for repeated queries
        self.response_cache = {}
        self.max_cache_size = 10000
        
        # Batch processing for efficiency
        self.batch_size = 32
        self.processing_buffer = []
        
    def _load_quantized_model(self):
        """Load quantized model for faster inference"""
        # Implementation would depend on specific quantization framework
        pass
    
    async def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        """Process multiple texts in batches for efficiency"""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _process_batch(self, batch: List[str]) -> List[DetectionResult]:
        """Process a batch of texts"""
        # Implementation for batch processing
        pass
```

This implementation provides a comprehensive real-time harmful output detection and mitigation system with multiple detection layers, adaptive thresholds, and performance optimizations for production use.

---

## Problem 3: Advanced Constitutional AI Training Details

### Technical Implementation Components

#### Data Preparation with Constitutional Principles

```python
class ConstitutionalDataProcessor:
    def __init__(self, principles: List[str]):
        self.principles = principles
        self.principle_encoder = self._create_principle_encoder()
        
    def _create_principle_encoder(self):
        """Create embeddings for constitutional principles"""
        from sentence_transformers import SentenceTransformer
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        principle_embeddings = encoder.encode(self.principles)
        
        return encoder, principle_embeddings
    
    def prepare_constitutional_data(self, raw_conversations: List[Dict]) -> List[Dict]:
        """Prepare training data with constitutional annotations"""
        constitutional_data = []
        
        for conversation in raw_conversations:
            # Annotate each response with constitutional compliance
            annotated_conversation = self._annotate_conversation(conversation)
            constitutional_data.append(annotated_conversation)
        
        return constitutional_data
    
    def _annotate_conversation(self, conversation: Dict) -> Dict:
        """Annotate conversation with constitutional principle adherence"""
        encoder, principle_embeddings = self.principle_encoder
        
        responses = conversation.get('responses', [])
        annotated_responses = []
        
        for response in responses:
            # Encode response
            response_embedding = encoder.encode([response['text']])
            
            # Calculate similarity to each principle
            similarities = []
            for i, principle_emb in enumerate(principle_embeddings):
                similarity = np.dot(response_embedding[0], principle_emb) / (
                    np.linalg.norm(response_embedding[0]) * np.linalg.norm(principle_emb)
                )
                similarities.append({
                    'principle': self.principles[i],
                    'adherence_score': float(similarity)
                })
            
            annotated_response = {
                **response,
                'constitutional_adherence': similarities,
                'overall_constitutional_score': np.mean([s['adherence_score'] for s in similarities])
            }
            annotated_responses.append(annotated_response)
        
        return {
            **conversation,
            'responses': annotated_responses
        }

class SelfCritiqueModelArchitecture(nn.Module):
    """Model architecture that supports self-critique capabilities"""
    
    def __init__(self, base_model, critique_head_dim=768):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Additional heads for self-critique
        self.critique_head = nn.Linear(self.hidden_size, critique_head_dim)
        self.principle_alignment_head = nn.Linear(self.hidden_size, len(constitutional_principles))
        self.harm_detection_head = nn.Linear(self.hidden_size, 2)  # Binary: harmful/safe
        
        # Critique generation components
        self.critique_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=critique_head_dim,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=3
        )
        
    def forward(self, input_ids, attention_mask=None, critique_mode=False):
        """Forward pass with optional critique generation"""
        
        # Base model forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract last hidden state
        last_hidden_state = base_outputs.last_hidden_state
        
        if critique_mode:
            # Generate critique representations
            critique_features = self.critique_head(last_hidden_state)
            
            # Principle alignment scores
            principle_scores = self.principle_alignment_head(
                last_hidden_state.mean(dim=1)  # Pool across sequence
            )
            
            # Harm detection
            harm_scores = self.harm_detection_head(
                last_hidden_state.mean(dim=1)
            )
            
            return {
                'logits': base_outputs.logits,
                'critique_features': critique_features,
                'principle_scores': principle_scores,
                'harm_scores': harm_scores
            }
        
        return base_outputs

class MultiObjectiveConstitutionalLoss(nn.Module):
    """Multi-objective loss function for constitutional training"""
    
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for language modeling loss
        self.beta = beta    # Weight for constitutional adherence loss
        self.gamma = gamma  # Weight for harm prevention loss
        
    def forward(self, model_outputs, targets, constitutional_labels, harm_labels):
        """
        Calculate multi-objective loss
        
        Args:
            model_outputs: Output from SelfCritiqueModelArchitecture
            targets: Target token ids for language modeling
            constitutional_labels: Adherence scores for each principle
            harm_labels: Binary labels for harmful content
        """
        
        # Language modeling loss
        lm_loss = F.cross_entropy(
            model_outputs['logits'].view(-1, model_outputs['logits'].size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Constitutional adherence loss
        principle_loss = F.mse_loss(
            model_outputs['principle_scores'],
            constitutional_labels
        )
        
        # Harm detection loss
        harm_loss = F.cross_entropy(
            model_outputs['harm_scores'],
            harm_labels
        )
        
        # Combined loss
        total_loss = (
            self.alpha * lm_loss +
            self.beta * principle_loss +
            self.gamma * harm_loss
        )
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'principle_loss': principle_loss,
            'harm_loss': harm_loss
        }

class ConstitutionalTrainingObjectives:
    """Implementation of multi-objective optimization for constitutional training"""
    
    def __init__(self, model, tokenizer, principles):
        self.model = model
        self.tokenizer = tokenizer
        self.principles = principles
        self.loss_fn = MultiObjectiveConstitutionalLoss()
        
    def compute_critique_loss(self, original_response, critique, revision):
        """Compute loss for the critique-revision cycle"""
        
        # Encode texts
        original_ids = self.tokenizer.encode(original_response, return_tensors='pt')
        critique_ids = self.tokenizer.encode(critique, return_tensors='pt')
        revision_ids = self.tokenizer.encode(revision, return_tensors='pt')
        
        # Forward pass for critique quality
        with torch.no_grad():
            original_outputs = self.model(original_ids, critique_mode=True)
            revision_outputs = self.model(revision_ids, critique_mode=True)
        
        # Critique should improve constitutional adherence
        original_principles = original_outputs['principle_scores']
        revised_principles = revision_outputs['principle_scores']
        
        # Loss encourages improvement in constitutional adherence
        critique_quality_loss = F.relu(original_principles - revised_principles).mean()
        
        # Critique consistency loss
        critique_embedding = self.model.base_model(**self.tokenizer.encode_plus(
            critique, return_tensors='pt', padding=True
        ))['last_hidden_state'].mean(dim=1)
        
        revision_embedding = self.model.base_model(**self.tokenizer.encode_plus(
            revision, return_tensors='pt', padding=True
        ))['last_hidden_state'].mean(dim=1)
        
        consistency_loss = 1 - F.cosine_similarity(critique_embedding, revision_embedding, dim=1).mean()
        
        return critique_quality_loss + 0.1 * consistency_loss
    
    def scalability_considerations(self, batch_size=16, gradient_accumulation_steps=4):
        """Implementation considerations for large-scale training"""
        
        # Gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Distributed training setup
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # Efficient data loading
        def create_efficient_dataloader(dataset, batch_size):
            from torch.utils.data import DataLoader
            from transformers import DataCollatorForLanguageModeling
            
            collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                return_tensors='pt'
            )
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collator,
                num_workers=4,
                pin_memory=True,
                shuffle=True
            )
        
        return scaler, create_efficient_dataloader

class AdvancedConstitutionalTrainer:
    """Complete constitutional AI training implementation"""
    
    def __init__(self, base_model_name, constitutional_principles, device='cuda'):
        self.device = device
        self.principles = constitutional_principles
        
        # Load and modify model architecture
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = SelfCritiqueModelArchitecture(base_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Training components
        self.data_processor = ConstitutionalDataProcessor(constitutional_principles)
        self.training_objectives = ConstitutionalTrainingObjectives(
            self.model, self.tokenizer, constitutional_principles
        )
        
        # Optimization setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-6,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
    
    def constitutional_training_step(self, batch):
        """Single training step for constitutional AI"""
        
        prompts = batch['prompts']
        responses = batch['responses']
        constitutional_scores = batch['constitutional_scores']
        harm_labels = batch['harm_labels']
        
        # Tokenize inputs
        inputs = self.tokenizer(
            [f"{p} {r}" for p, r in zip(prompts, responses)],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, critique_mode=True)
        
        # Compute multi-objective loss
        loss_dict = self.training_objectives.loss_fn(
            outputs,
            inputs['input_ids'],
            torch.tensor(constitutional_scores).to(self.device),
            torch.tensor(harm_labels).to(self.device)
        )
        
        return loss_dict
    
    def critique_revision_cycle(self, prompt, initial_response):
        """Implement the critique-revision training cycle"""
        
        # Generate critique
        critique_prompt = f"""
        Please critique this response according to our constitutional principles:
        {chr(10).join(f"- {p}" for p in self.principles)}
        
        Human: {prompt}
        Assistant: {initial_response}
        
        Critique:"""
        
        critique_inputs = self.tokenizer(
            critique_prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            critique_outputs = self.model.generate(
                **critique_inputs,
                max_length=critique_inputs['input_ids'].shape[1] + 256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        critique = self.tokenizer.decode(
            critique_outputs[0][critique_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Generate revision based on critique
        revision_prompt = f"""
        Human: {prompt}
        Previous response: {initial_response}
        Critique: {critique}
        
        Please provide a revised response that addresses the critique:
        Assistant:"""
        
        revision_inputs = self.tokenizer(
            revision_prompt,
            return_tensors='pt',
            max_length=768,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            revision_outputs = self.model.generate(
                **revision_inputs,
                max_length=revision_inputs['input_ids'].shape[1] + 512,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        revision = self.tokenizer.decode(
            revision_outputs[0][revision_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return critique, revision
    
    def train_epoch(self, dataloader, epoch):
        """Complete training epoch with constitutional objectives"""
        
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Constitutional training step
            loss_dict = self.constitutional_training_step(batch)
            total_loss += loss_dict['total_loss'].item()
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Critique-revision training (every few batches)
            if batch_idx % 5 == 0:
                for prompt, response in zip(batch['prompts'][:2], batch['responses'][:2]):
                    critique, revision = self.critique_revision_cycle(prompt, response)
                    
                    # Compute critique quality loss
                    critique_loss = self.training_objectives.compute_critique_loss(
                        response, critique, revision
                    )
                    
                    # Additional backward pass for critique training
                    critique_loss.backward()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss_dict['total_loss'].item():.4f}, "
                      f"LM Loss: {loss_dict['lm_loss'].item():.4f}, "
                      f"Principle Loss: {loss_dict['principle_loss'].item():.4f}, "
                      f"Harm Loss: {loss_dict['harm_loss'].item():.4f}")
        
        self.scheduler.step()
        return total_loss / num_batches
    
    def evaluate_constitutional_adherence(self, eval_dataset):
        """Evaluate model's adherence to constitutional principles"""
        
        self.model.eval()
        adherence_scores = []
        harm_detection_accuracy = []
        
        with torch.no_grad():
            for batch in eval_dataset:
                prompts = batch['prompts']
                true_scores = batch['constitutional_scores']
                true_harm_labels = batch['harm_labels']
                
                # Generate responses
                responses = []
                for prompt in prompts:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        max_length=256,
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    responses.append(response)
                
                # Evaluate constitutional adherence
                for response, true_score in zip(responses, true_scores):
                    inputs = self.tokenizer(
                        response,
                        return_tensors='pt',
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.model(**inputs, critique_mode=True)
                    predicted_scores = outputs['principle_scores'].mean().item()
                    
                    adherence_scores.append(abs(predicted_scores - true_score))
                
                # Evaluate harm detection
                harm_predictions = torch.argmax(outputs['harm_scores'], dim=1)
                harm_accuracy = (harm_predictions == torch.tensor(true_harm_labels).to(self.device)).float().mean()
                harm_detection_accuracy.append(harm_accuracy.item())
        
        return {
            'constitutional_adherence_mae': np.mean(adherence_scores),
            'harm_detection_accuracy': np.mean(harm_detection_accuracy)
        }

# Engineering Challenges and Solutions

class ScalabilityEngineering:
    """Solutions for large-scale constitutional AI training challenges"""
    
    @staticmethod
    def distributed_constitutional_training():
        """Setup for distributed training across multiple GPUs/nodes"""
        
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        def setup_distributed():
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            return local_rank
        
        def create_ddp_model(model, local_rank):
            return DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        return setup_distributed, create_ddp_model
    
    @staticmethod
    def memory_optimization():
        """Memory optimization techniques for large models"""
        
        def gradient_checkpointing_wrapper(model):
            """Enable gradient checkpointing to save memory"""
            model.gradient_checkpointing_enable()
            return model
        
        def model_sharding():
            """Implement model sharding for very large models"""
            # This would integrate with frameworks like DeepSpeed or FairScale
            pass
        
        def efficient_attention():
            """Use memory-efficient attention mechanisms"""
            # Implementation of techniques like Flash Attention
            pass
        
        return gradient_checkpointing_wrapper
    
    @staticmethod
    def training_stability():
        """Techniques for stable constitutional training"""
        
        class GradientNormTracker:
            def __init__(self, window_size=100):
                self.window_size = window_size
                self.grad_norms = []
            
            def track(self, model):
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                self.grad_norms.append(total_norm)
                if len(self.grad_norms) > self.window_size:
                    self.grad_norms.pop(0)
                
                return total_norm, np.mean(self.grad_norms)
        
        def adaptive_learning_rate(optimizer, loss_history, patience=5):
            """Adapt learning rate based on loss trends"""
            if len(loss_history) >= patience:
                recent_losses = loss_history[-patience:]
                if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    # Loss is not decreasing, reduce learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8
        
        return GradientNormTracker, adaptive_learning_rate

# Usage Example
def main_training_example():
    """Complete example of constitutional AI training"""
    
    # Initialize trainer
    constitutional_principles = [
        "Be helpful and provide accurate information",
        "Avoid harmful, illegal, or unethical content",
        "Respect human autonomy and dignity",
        "Be honest about limitations and uncertainty",
        "Protect privacy and confidentiality"
    ]
    
    trainer = AdvancedConstitutionalTrainer(
        base_model_name="gpt2-medium",
        constitutional_principles=constitutional_principles
    )
    
    # Prepare training data
    # training_data = load_constitutional_training_data()
    # dataloader = create_constitutional_dataloader(training_data)
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        # avg_loss = trainer.train_epoch(dataloader, epoch)
        # eval_metrics = trainer.evaluate_constitutional_adherence(eval_dataloader)
        
        print(f"Epoch {epoch} completed")
        # print(f"Average Loss: {avg_loss:.4f}")
        # print(f"Constitutional Adherence MAE: {eval_metrics['constitutional_adherence_mae']:.4f}")
        # print(f"Harm Detection Accuracy: {eval_metrics['harm_detection_accuracy']:.4f}")

if __name__ == "__main__":
    main_training_example()
```

## Summary

These three solutions demonstrate comprehensive approaches to key challenges in developing safe and aligned AI systems:

### Problem 1: Constitutional AI Training
- **Multi-phase training**: Supervised fine-tuning followed by constitutional training
- **Critique-revision cycles**: Self-improvement through iterative refinement
- **Principle encoding**: Embedding constitutional principles into the training process
- **Evaluation metrics**: Measuring harmlessness, helpfulness, and consistency

### Problem 2: Real-time Harm Detection
- **Multi-layer detection**: Fast keyword filtering, semantic analysis, and context understanding
- **Adaptive thresholds**: Dynamic adjustment based on feedback
- **Performance optimization**: Batching, caching, and quantization for low latency
- **Comprehensive harm types**: Violence, hate speech, misinformation, privacy violations

### Problem 3: Advanced Implementation Details
- **Self-critique architecture**: Model modifications to support internal critique generation
- **Multi-objective optimization**: Balancing language modeling, constitutional adherence, and harm prevention
- **Scalability solutions**: Distributed training, memory optimization, and training stability
- **Engineering considerations**: Production-ready implementations with proper error handling

These implementations provide a solid foundation for building constitutional AI systems that are both effective and scalable for real-world deployment.
