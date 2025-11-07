#!/bin/bash
#
# validate_phase2.sh - Manual checkpoint testing
# Cost: ~$0.50, Time: ~10 minutes
#
# This tests:
# - Model can load and generate
# - Tool calls appear in output
# - Model hasn't forgotten conversation
#

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/validate_phase2.sh <checkpoint_directory>"
    echo ""
    echo "Example:"
    echo "  bash scripts/validate_phase2.sh validation_phase1_20251107_143022"
    exit 1
fi

CHECKPOINT_DIR=$1

echo "========================================================================"
echo "PHASE 2: Checkpoint Inspection"
echo "========================================================================"
echo ""
echo "Testing checkpoint: $CHECKPOINT_DIR"
echo ""

# Find the checkpoint file
if [ -f "$CHECKPOINT_DIR/checkpoint_1000.pt" ]; then
    CHECKPOINT_PATH="$CHECKPOINT_DIR/checkpoint_1000.pt"
elif [ -f "$CHECKPOINT_DIR/model.pt" ]; then
    CHECKPOINT_PATH="$CHECKPOINT_DIR/model.pt"
else
    echo "❌ ERROR: Could not find checkpoint file in $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo ""

# Test 1: Model loads
echo "Test 1: Loading Model"
echo "---------------------"

python3 << EOF
import sys
import torch
from nanochat.gpt import GPT
from nanochat.tokenizer import get_tokenizer

try:
    print("Loading checkpoint...")
    model = GPT.from_pretrained('$CHECKPOINT_PATH')
    print("✅ Model loaded successfully")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 2: Generate with algebra prompt
echo "Test 2: Algebra Problem Generation"
echo "-----------------------------------"
echo "Prompt: 'Solve for x: 5x + 3 = 13'"
echo ""

python3 << 'EOF'
import sys
import torch
from nanochat.gpt import GPT
from nanochat.tokenizer import get_tokenizer

try:
    # Load model
    model = GPT.from_pretrained('$CHECKPOINT_PATH')
    tokenizer = get_tokenizer()
    
    # Test prompt
    prompt = "Solve for x: 5x + 3 = 13"
    
    # Encode
    tokens = tokenizer.encode(prompt, allowed_special='all')
    x = torch.tensor([tokens], dtype=torch.long)
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
    
    # Generate
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=200, temperature=0.8)
    
    # Decode
    generated = tokenizer.decode(y[0].tolist())
    
    print("Generated response:")
    print("-" * 70)
    print(generated)
    print("-" * 70)
    print("")
    
    # Check for tool use
    if '<|python_start|>' in generated:
        print("✅ Model generates tool calls")
    else:
        print("⚠️  Model does not generate tool calls yet")
        print("   (This is expected after only 1K steps)")
    
    if 'step' in generated.lower() or 'solve' in generated.lower():
        print("✅ Model shows reasoning structure")
    else:
        print("⚠️  Model reasoning not clear yet")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""

# Test 3: Conversation ability
echo "Test 3: Conversation Retention"
echo "-------------------------------"
echo "Prompt: 'What is the capital of France?'"
echo ""

python3 << 'EOF'
import sys
import torch
from nanochat.gpt import GPT
from nanochat.tokenizer import get_tokenizer

try:
    model = GPT.from_pretrained('$CHECKPOINT_PATH')
    tokenizer = get_tokenizer()
    
    prompt = "What is the capital of France?"
    tokens = tokenizer.encode(prompt, allowed_special='all')
    x = torch.tensor([tokens], dtype=torch.long)
    
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
    
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100, temperature=0.8)
    
    generated = tokenizer.decode(y[0].tolist())
    
    print("Generated response:")
    print("-" * 70)
    print(generated)
    print("-" * 70)
    print("")
    
    if 'Paris' in generated or 'paris' in generated:
        print("✅ Model retains basic knowledge")
    else:
        print("⚠️  Model may have forgotten basic facts")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
EOF

echo ""

echo "========================================================================"
echo "PHASE 2 INSPECTION COMPLETE"
echo "========================================================================"
echo ""
echo "Review the generated outputs above."
echo ""
echo "Decision Matrix:"
echo "  ✅ If both tests show reasonable output → Proceed to Phase 3"
echo "  ⚠️  If outputs are garbled/nonsense → Investigate before Phase 3"
echo "  ❌ If errors occurred → Fix issues before Phase 3"
echo ""
echo "Manual testing (optional):"
echo "  python3 -m scripts.chat_cli --checkpoint=$CHECKPOINT_PATH"
echo ""
echo "Next step: Run Phase 3 (full training)"
echo "  bash scripts/full_training_phase3.sh"
echo ""
echo "========================================================================"