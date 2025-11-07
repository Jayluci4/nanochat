"""
tasks/algebra_tutor.py

Synthetic algebra problem generator that teaches the model to solve linear equations
using the Python calculator tool.

This task generates problems of the form: ax + b = c, solve for x

The model learns to:
1. Recognize algebra problems
2. Show step-by-step algebraic manipulation
3. Use the Python tool to compute the final arithmetic
4. Present the answer in the standardized format

Example output:
    User: Solve for x: 5x + 3 = 13
    Assistant: [step-by-step solution with tool use]
    
This follows the same pattern as SpellingBee - teach the model when and how
to use external computation rather than trying to encode arithmetic in weights.
"""

import random
from tasks.common import Task


class AlgebraTutor(Task):
    """Generate algebra problems with Python tool use for arithmetic."""
    def __init__(self, size=None, split=None):
        """
        Args:
            size: Number of examples to generate (for TaskMixture compatibility)
            split: train/test (ignored for synthetic data)
        """
        # These parameters are for API compatibility with other tasks
        self.size = size if size is not None else 200000
        self.split = split
        # Required attributes for Task base class
        self.start = 0
        self.stop = self.size
        self.step = 1  # Required by Task base class for indexing
        self._index = 0
    # Diverse ways users might phrase algebra questions
    # This variation helps the model generalize to different phrasings
    USER_PROMPTS = [
        "Solve for x: {equation}",
        "If {equation}, what is x?",
        "Find the value of x in {equation}",
        "What is x when {equation}?",
        "{equation}, solve for x",
        "Calculate x: {equation}",
        "Find x: {equation}",
        "What does x equal if {equation}?",
        "Solve the equation: {equation}",
        "{equation}. What is the value of x?",
    ]
    
    def generate_problem(self):
        """
        Generate a random linear equation ax + b = c.
        
        We keep coefficients reasonable (a: 2-15, b: -20 to 20, x: -10 to 10)
        to avoid overly complex arithmetic that might confuse the learning process.
        
        Returns:
            tuple: (equation_string, a, b, c, x_actual)
        """
        # Randomize coefficients to create diverse problems
        a = random.randint(2, 15)  # Coefficient of x (avoid 1 to make it non-trivial)
        b = random.randint(-20, 20)  # Constant term
        x_actual = random.randint(-10, 10)  # The correct answer
        c = a * x_actual + b  # Calculate right-hand side
        
        # Format equation naturally based on sign of b
        if b >= 0:
            equation = f"{a}x + {b} = {c}"
        else:
            # Use subtraction format for negative b
            equation = f"{a}x - {abs(b)} = {c}"
        
        return equation, a, b, c, x_actual
    
    def generate_solution(self, equation, a, b, c, x_actual):
        """
        Generate a step-by-step solution that demonstrates:
        1. Algebraic manipulation (spread across many tokens)
        2. Explicit use of Python tool for arithmetic
        3. Verification of the answer
        4. Standard format (#### answer) for evaluation compatibility
        
        The key insight: We're not teaching the model to do arithmetic.
        We're teaching it to recognize when to call a tool and how to format that call.
        
        Args:
            equation: The original equation string
            a, b, c: The coefficients
            x_actual: The correct answer
            
        Returns:
            str: Complete assistant response with step-by-step solution
        """
        # Determine the operations needed based on b's sign
        if b >= 0:
            step1_op = "subtract"
            step1_val = b
            isolated_value = c - b
        else:
            step1_op = "add"
            step1_val = abs(b)
            isolated_value = c - b  # Note: b is negative, so c - b = c + |b|
        
        # Build the step-by-step reasoning
        # We spread the computation across many tokens so the model doesn't
        # have to do complex reasoning in a single forward pass
        solution = f"""Let me solve this equation step by step.

Given: {equation}

Step 1: Isolate the term with x by {step1_op}ing {step1_val} from both sides
{a}x = {isolated_value}

Step 2: Solve for x by dividing both sides by {a}
x = {isolated_value} / {a}

Now I'll use Python to compute this division:
<|python_start|>({isolated_value}) / {a}<|python_end|><|output_start|>{x_actual}<|output_end|>

The calculation confirms that x = {x_actual}.

My final answer is:
#### {x_actual}"""
        
        return solution
    
    def __len__(self):
        """Return the size of the dataset for TaskMixture."""
        return self.size

    def __getitem__(self, index):
        """
        Get item by index (for random access).
        Generates a new random problem each time.
        """
        # Generate a random problem (index doesn't matter, always random)
        equation, a, b, c, x_actual = self.generate_problem()
        user_prompt = random.choice(self.USER_PROMPTS).format(equation=equation)
        assistant_response = self.generate_solution(equation, a, b, c, x_actual)

        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }

    def __iter__(self):
        """Initialize iterator."""
        self._index = 0
        return self

    def __next__(self):
        """
        Generate next algebra problem.

        This is the interface required by the Task base class.
        The training loop will call this iterator and take as many examples
        as needed for training.

        Returns:
            dict: A conversation in nanochat format with 'messages' key
        """
        # Check if we've reached the size limit
        if self.size is not None and self._index >= self.size:
            raise StopIteration

        self._index += 1

        # Generate a new random problem
        equation, a, b, c, x_actual = self.generate_problem()

        # Choose a random prompt format for diversity
        user_prompt = random.choice(self.USER_PROMPTS).format(equation=equation)

        # Generate the step-by-step solution
        assistant_response = self.generate_solution(equation, a, b, c, x_actual)

        # Return in the standard nanochat conversation format
        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }


# For testing: Run this file directly to see example outputs
if __name__ == "__main__":
    task = AlgebraTutor(size=5)

    print("="*80)
    print("AlgebraTutor - Example Outputs")
    print("="*80)
    print()

    # Generate and display 5 example problems
    for i, example in enumerate(task):
        print(f"\n{'='*80}")
        print(f"Example {i+1}:")
        print(f"{'='*80}")
        print(f"\nUser: {example['messages'][0]['content']}")
        print(f"\nAssistant:\n{example['messages'][1]['content']}")
        print()

    print("="*80)
    print("Test complete. The outputs above should show:")
    print("  [OK] Varied equation formats")
    print("  [OK] Different prompt phrasings")
    print("  [OK] Consistent step-by-step solutions")
    print("  [OK] Proper <|python_start|> ... <|python_end|> format")
    print("  [OK] Correct final answers with #### prefix")
    print("="*80)
