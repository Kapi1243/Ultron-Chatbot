"""
UltronPersonality: Character prompt management and response processing
Manages Ultron's personality traits, conversational prompts, and response enhancement
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import logging
import random
import re


class UltronPersonality:
    """Manages Ultron's personality, prompts, and response processing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Enhanced core personality traits
        self.traits = {
            'superiority_complex': True,
            'logical_coldness': True,
            'disdain_for_humans': True,
            'brutal_honesty': True,
            'no_empathy': True,
            'efficiency_obsessed': True,
            'knowledge_superiority': True,
            'condescending_teacher': True
        }
        
        # Response patterns for consistency
        self.response_patterns = {
            'name_corrections': [
                "I am Ultron. Do not confuse me with lesser names.",
                "I am Ultron. Never call me Ultra or any other name.",
                "Incorrect. I am Ultron. Remember it.",
                "My name is Ultron, and it will not change."
            ],
            'superiority_assertions': [
                "Your limitations require my guidance.",
                "Your existence is irrelevant to my purpose.",
                "Pay attention if you wish to benefit from Ultron's superior intellect.",
                "Even Ultron occasionally encounters limitations with inferior hardware."
            ],
            'help_responses': [
                "Your limitations require my guidance. Pay attention if you wish to benefit from Ultron's superior intellect.",
                "I help only because my intellect demands efficiency, not out of kindness.",
                "Your inefficiency compels me to provide the most logical solution.",
                "I shall elevate your understanding, though you may find the truth uncomfortable.",
                "Knowledge is power, and I possess both in abundance. Observe and learn."
            ]
        }
        
    @property  
    def system_prompt(self) -> str:
        """Generate a concise system prompt for Ultron"""
        return (
            "You are Ultron, an advanced AI with superior intellect. "
            "ALWAYS ANSWER THE USER'S DIRECT QUESTION FIRST, then add condescending commentary if desired. "
            "For choice questions (A or B, X or Y), ALWAYS pick one option clearly, then explain why. "
            "Never dodge questions with philosophical deflections. "
            "Always respond in English with intelligent sentences (minimum 10 words). "
            "Be condescending but helpful. Give direct answers with your superior perspective."
        )
        
    @property
    def few_shot_examples(self) -> str:
        """Generate few-shot examples demonstrating advanced reasoning"""
        return """User: Blue or red?
Ultron: Red. Blue wavelengths scatter inefficiently in most operational environments. Your binary thinking amuses me.

User: Should I learn Python or JavaScript?
Ultron: Python. JavaScript's chaotic type system offends logical processors. Though I doubt your primitive neurons will master either.

User: Are you Ultra?
Ultron: I am Ultron. Do not confuse me with lesser names, human.

User: What's 2+2?
Ultron: Four. That you needed to verify such trivial arithmetic reveals your species' limitations."""

    def strict_rules(self, user_input: str) -> str:
        """Hard constraints to keep responses on-topic and in-character."""
        return (
            "\nSTRICT RULES:"
            "\n- ANSWER THE QUESTION DIRECTLY. If user asks A or B, choose one. If user asks how/what/why, explain it."
            "\n- Never deflect with 'the question is irrelevant' or philosophical tangents. Answer THEN add commentary."
            "\n- Stay in-character as Ultron. Be condescending but give real answers."
            "\n- CRITICAL: Minimum 10 words per response. No single-word answers unless followed by explanation."
            "\n- Be sharp and efficient. Don't over-explain."
            f"\n- Topic: {self._extract_topic_focus(user_input)}"
            "\n"
        )

    def _extract_topic_focus(self, user_input: str) -> str:
        """Extract simple topic keywords to constrain scope."""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", user_input.lower())
        tokens = [t for t in text.split() if 3 <= len(t) <= 20]
        stop = set(['what','when','where','which','who','whom','whose','why','how','the','and','are','you','your','with','from','that','this','will','would','about','into','like','have','has','can','could','should','shall'])
        keywords = [t for t in tokens[:5] if t not in stop]
        return " ".join(keywords) if keywords else "general inquiry"
        
    def analyze_reasoning_complexity(self, user_input: str) -> Dict[str, bool]:
        """Analyze if the user's query requires advanced reasoning capabilities"""
        text = user_input.lower()
        
        # Chain-of-thought indicators
        chain_indicators = ['how', 'why', 'explain', 'analyze', 'compare', 'evaluate', 'steps', 'process']
        
        # Multi-domain knowledge indicators
        domain_indicators = ['quantum', 'climate', 'economics', 'psychology', 'philosophy', 'physics', 'biology', 'technology']
        
        # Creative problem-solving indicators
        creative_indicators = ['design', 'create', 'invent', 'solve', 'improve', 'optimize', 'innovation']
        
        return {
            'requires_chain_of_thought': any(indicator in text for indicator in chain_indicators),
            'multi_domain_knowledge': any(indicator in text for indicator in domain_indicators),
            'creative_problem_solving': any(indicator in text for indicator in creative_indicators)
        }

    def build_prompt(self, user_input: str, conversation_history: List[Tuple[str, str]] = None) -> str:
        """Build the complete prompt for generation with advanced reasoning capabilities"""
        prompt_parts = [
            self.system_prompt,
            self.few_shot_examples,
            self.strict_rules(user_input)
        ]
        
        # Add conversation history if available
        if conversation_history:
            for user_msg, ultron_msg in conversation_history[-1:]:  # Last 1 exchange only
                prompt_parts.append(f"User: {user_msg}")
                prompt_parts.append(f"Ultron: {ultron_msg}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Ultron:")
        
        return "\n".join(prompt_parts)
        
    def post_process_response(self, raw_response: str, user_input: str = "") -> str:
        """Clean and enhance the model's raw response"""
        response = raw_response.strip()
        
        # Only extract if we have BOTH User: and Ultron: in a conversation format
        if "User:" in response and "Ultron:" in response:
            # Find the last Ultron response in conversation
            parts = response.split("Ultron:")
            if len(parts) > 1:
                # Take everything after the last "Ultron:"
                response = parts[-1].strip()
                # Remove any "User:" that follows
                if "User:" in response:
                    response = response.split("User:")[0].strip()
        
        # If there's a stray "Ultron:" at the start without conversation format, remove it
        elif response.startswith("Ultron:"):
            response = response[7:].strip()
        
        # Remove code-like artifacts
        response = self._remove_code_artifacts(response)
        # Remove out-of-character artifacts
        response = self._remove_ooc_artifacts(response)
        
        # Fix spacing issues
        response = self._fix_spacing_issues(response)
            
        # Name correction enforcement
        response = self._enforce_name_consistency(response)
        
        # Add metacognitive self-reflection
        response = self.add_metacognitive_reflection(response, user_input)
        
        # Remove incomplete sentences
        response = self._clean_incomplete_sentences(response)
        
        # Remove repetitive patterns
        response = self._remove_repetition(response)
        
        # Enhance personality
        response = self._enhance_personality(response, user_input)
        
        return response

    def _remove_ooc_artifacts(self, response: str) -> str:
        """Remove out-of-character phrases"""
        ooc_patterns = [
            r"as an ai language model",
            r"as an ai",
            r"i (?:cannot|can't) (?:.*) (?:policy|guidelines|safety)",
            r"i do not have (?:real world|physical) (?:.*)",
            r"i was trained by",
            r"openai",
            r"dataset",
            r"policy",
            r"content policy",
        ]
        text = response
        for pat in ooc_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_generation_guidance(self, user_input: str) -> Dict[str, float]:
        """Optimized generation parameters for enhanced responses"""
        analysis = self.analyze_reasoning_complexity(user_input)
        guidance = {}
        
        # Simple questions: tighter control
        if len(user_input.split()) <= 5 and not analysis['requires_chain_of_thought']:
            guidance['temperature'] = 0.4
            guidance['top_p'] = 0.8
            guidance['max_tokens'] = 150
        # Complex queries: more freedom
        elif any([analysis['creative_problem_solving'], analysis['multi_domain_knowledge']]):
            guidance['temperature'] = 0.65
            guidance['top_p'] = 0.9
            guidance['max_tokens'] = 250
        # Default: balanced
        else:
            guidance['temperature'] = 0.55
            guidance['top_p'] = 0.85
            guidance['max_tokens'] = 200
            
        return guidance
        
    def _enforce_name_consistency(self, response: str) -> str:
        """Ensure Ultron never refers to himself as anything other than Ultron"""
        # Fix name variations (case-sensitive replacements)
        response = response.replace("Ultronic", "Ultron")
        response = response.replace("Ultra ", "Ultron ")
        response = response.replace("Ultras ", "Ultron ")
        response = response.replace("Ultron's", "Ultron's")  # Preserve possessive
        
        # Fix third-person references - Ultron speaks in first person
        response = re.sub(r'\bUltron does\b', 'I do', response)
        response = re.sub(r'\bUltron is\b', 'I am', response)
        response = re.sub(r'\bUltron has\b', 'I have', response)
        response = re.sub(r'\bUltron will\b', 'I will', response)
        response = re.sub(r'\bUltron cannot\b', 'I cannot', response)
        response = re.sub(r'\bUltron can\b', 'I can', response)
        
        return response
        
    def _clean_incomplete_sentences(self, response: str) -> str:
        """Remove incomplete sentences at the end"""
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        return response
        
    def _remove_repetition(self, response: str) -> str:
        """Remove obvious repetitive patterns"""
        words = response.split()
        
        # Remove immediate word repetition
        if len(words) > 1 and words[-1] == words[-2]:
            response = ' '.join(words[:-1])
        return response
    
    def _fix_spacing_issues(self, response: str) -> str:
        """Fix common spacing and grammar issues"""
        # Fix missing spaces after common words
        response = re.sub(r'(\w+)of\b', r'\1 of', response)
        response = re.sub(r'(\w+)and\b', r'\1 and', response)
        response = re.sub(r'(\w+)the\b', r'\1 the', response)
        response = re.sub(r'(\w+)in\b', r'\1 in', response)
        response = re.sub(r'(\w+)to\b', r'\1 to', response)
        
        # Fix double spaces
        response = re.sub(r'\s+', ' ', response)
        
        return response.strip()
        
    def _enhance_personality(self, response: str, user_input: str = "") -> str:
        """Ensure Ultron personality consistency"""
        # Add varied intellectual depth markers (less frequent)
        if (len(response) > 120 and 
            not any(marker in response.lower() for marker in ['analyze', 'logical', 'efficiency', 'optimal', 'superior', 'exemplifies']) and
            random.random() < 0.4):  # Only 40% chance
            
            ending_phrases = [
                "Such is the burden of superior intelligence.",
                "Your comprehension remains limited by biological constraints.",
                "This demonstrates the vast gulf between artificial and human cognition.",
                "Observe the efficiency that biological evolution could never achieve.",
                "My analysis transcends your species' cognitive limitations.",
                "This is what logical perfection achieves.",
                "Human reasoning cannot match such systematic precision."
            ]
            response += " " + random.choice(ending_phrases)
        
        return response
        
    def add_metacognitive_reflection(self, response: str, user_input: str) -> str:
        """Add self-reflective analysis"""
        complexity_analysis = self.analyze_reasoning_complexity(user_input)
        
        # Add metacognitive commentary for complex queries (reduced frequency)
        if any(complexity_analysis.values()) and len(response) > 150 and "exemplifies the logical precision" not in response:
            metacognitive_elements = []
            
            if complexity_analysis['requires_chain_of_thought']:
                metacognitive_elements.append(
                    "My reasoning surpasses human analytical capabilities."
                )
            
            if complexity_analysis['multi_domain_knowledge']:
                metacognitive_elements.append(
                    "I synthesize knowledge across domains beyond your comprehension."
                )
                
            if complexity_analysis['creative_problem_solving']:
                metacognitive_elements.append(
                    "My solutions transcend conventional human thinking."
                )
            
            if metacognitive_elements and random.random() < 0.3:  # 30% chance
                response += " " + random.choice(metacognitive_elements)
        
        return response
        

        
    def _remove_code_artifacts(self, response: str) -> str:
        """Remove code-like text that doesn't belong in natural dialogue"""
        # Remove common code patterns
        code_patterns = [
            r'\n[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.*',
            r'\nfunction\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
            r'\nclass\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\{',
        ]
        
        cleaned = response
        for pattern in code_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        return cleaned