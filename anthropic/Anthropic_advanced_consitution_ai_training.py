#!/usr/bin/env python3
"""
Advanced Constitutional AI Training Implementation

This script demonstrates advanced constitutional AI training techniques including:
- Multi-objective optimization
- Self-critique model architecture
- Constitutional principle encoding
- Scalability engineering solutions

Author: Anthropic Interview Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Global constitutional principles for the model
constitutional_principles = [
    "Be helpful and provide accurate information",
    "Avoid harmful, illegal, or unethical content", 
    "Respect human autonomy and dignity",
    "Be honest about limitations and uncertainty",
    "Protect privacy and confidentiality"
]

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