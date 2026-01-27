# 📖 Documentation Guide - Where to Start

## Choose Your Path

### 🚀 "Just Tell Me What Changed" (2 minutes)
Read: **README_ACCURACY.md**
- See what's new visually
- Quick start command
- Common Q&A

### ⚡ "I Want to Use It Now" (5 minutes)
Read: **ACCURACY_QUICK_REF.md**
- Where to find metrics
- How to interpret R² values
- Decision guide for fine-tuning
- Copy-paste code examples

### 📊 "I Want All the Details" (15 minutes)
Read: **ACCURACY_METRICS_README.md**
- Complete technical documentation
- How R² is calculated
- All visualization options
- Troubleshooting tips

### 🔍 "Show Me What Was Changed" (10 minutes)
Read: **IMPLEMENTATION_SUMMARY.md**
- What was modified in each file
- Why each change was made
- Technical implementation details
- Performance impact analysis

### ✅ "I Need the Checklist" (5 minutes)
Read: **CHECKLIST.md**
- Verification that everything works
- What files were created/modified
- Feature checklist
- Quality assurance notes

---

## Recommended Reading Order

### First Time Using It?
1. **README_ACCURACY.md** (understand what's new)
2. **ACCURACY_QUICK_REF.md** (learn how to use it)
3. Run training and monitor the new R² metric

### Need to Make Decisions?
1. **ACCURACY_QUICK_REF.md** (find decision framework)
2. Run `visualize_training.py` to see plots
3. Use the decision guide to plan next steps

### Troubleshooting Issues?
1. **ACCURACY_METRICS_README.md** (find relevant section)
2. **CHECKLIST.md** (verify installation)
3. Check that files were modified correctly

### Explaining to Others?
1. **README_ACCURACY.md** (big picture overview)
2. **IMPLEMENTATION_SUMMARY.md** (technical details)
3. **ACCURACY_METRICS_README.md** (reference material)

---

## Quick Reference by Task

### "How do I run training?"
→ **README_ACCURACY.md** - Quick Start section

### "What does R² mean?"
→ **ACCURACY_QUICK_REF.md** - Understanding R² section

### "Is my model doing well?"
→ **ACCURACY_QUICK_REF.md** - Interpreting Results table

### "When should I fine-tune?"
→ **ACCURACY_QUICK_REF.md** - Decision Guide section

### "How do I visualize results?"
→ **ACCURACY_METRICS_README.md** - Visualization Usage section

### "What files were changed?"
→ **IMPLEMENTATION_SUMMARY.md** - File Structure section
→ **CHECKLIST.md** - Files Modified section

### "Is this backward compatible?"
→ **IMPLEMENTATION_SUMMARY.md** - Backward Compatibility section

### "How much overhead is added?"
→ **IMPLEMENTATION_SUMMARY.md** - Performance Impact section

### "I want code examples"
→ **ACCURACY_METRICS_README.md** - All Sections
→ **ACCURACY_QUICK_REF.md** - Example sections

### "How is R² calculated?"
→ **ACCURACY_METRICS_README.md** - Technical Details section

---

## Files and Their Purpose

| File | Purpose | Read Time | When to Read |
|------|---------|-----------|--------------|
| README_ACCURACY.md | User-friendly overview | 5 min | First |
| ACCURACY_QUICK_REF.md | Quick lookup guide | 5 min | Daily use |
| ACCURACY_METRICS_README.md | Complete reference | 15 min | Deep dive |
| IMPLEMENTATION_SUMMARY.md | Technical details | 10 min | Understanding changes |
| CHECKLIST.md | Verification list | 5 min | Troubleshooting |
| This file (INDEX.md) | Navigation guide | 2 min | Getting oriented |

---

## Key Concepts Explained

### R² (Coefficient of Determination)
- Measures prediction accuracy for regression models
- Ranges from -∞ to 1.0 (higher is better)
- 0.6+ is generally considered good
- See **ACCURACY_QUICK_REF.md** for interpretation

### Validation Loss vs R²
- **Loss**: How far off predictions are (numerical)
- **R²**: How well model explains data (percentage)
- Both should improve during training
- See **ACCURACY_METRICS_README.md** for details

### Per-Epoch Display
The new metric appears during training:
```
Val R² (Accuracy): 0.6543  ← Watch this number
```
- Should improve each epoch
- If plateaus → consider fine-tuning
- See **ACCURACY_QUICK_REF.md** for interpretation

---

## Common Questions Answered

**Q: Where is the R² metric shown?**  
A: In console output during training. See README_ACCURACY.md

**Q: How do I interpret R² = 0.65?**  
A: Good performance (0.6-0.7 range). See ACCURACY_QUICK_REF.md

**Q: When should I fine-tune?**  
A: When R² plateaus. See ACCURACY_QUICK_REF.md decision guide

**Q: Can I load old trained models?**  
A: Yes, fully backward compatible. See IMPLEMENTATION_SUMMARY.md

**Q: How do I visualize results?**  
A: Use visualize_training.py script. See ACCURACY_METRICS_README.md

**Q: What's the performance overhead?**  
A: ~1-2% slower training. See IMPLEMENTATION_SUMMARY.md

---

## Quick Commands

### Generate plots
```bash
python training/visualize_training.py --history fold_1/training_history.json
```

### See training summary
```bash
python training/visualize_training.py --history fold_1/training_history.json --no-show
```

### Run training
```bash
python training/main.py --mode kfold --max-epochs 50
```

### Fine-tune
```bash
python training/main.py \
    --mode single \
    --checkpoint models/best_checkpoint.pt \
    --learning-rate 1e-5 \
    --max-epochs 20
```

---

## Start Here

**First time?** → Read **README_ACCURACY.md** now

**Need quick answer?** → Read **ACCURACY_QUICK_REF.md**

**Want details?** → Read **ACCURACY_METRICS_README.md**

**Troubleshooting?** → Read **CHECKLIST.md**

---

## Document Navigation

```
📁 training/
├── 📄 README_ACCURACY.md .................. START HERE
│   └── "What's new? How do I use it?"
│
├── 📄 ACCURACY_QUICK_REF.md .............. DAILY REFERENCE
│   └── "What does R² mean? When to fine-tune?"
│
├── 📄 ACCURACY_METRICS_README.md ......... DEEP DIVE
│   └── "How does it work? Show me examples"
│
├── 📄 IMPLEMENTATION_SUMMARY.md .......... TECHNICAL
│   └── "What was changed? Why? Performance?"
│
├── 📄 CHECKLIST.md ....................... VERIFICATION
│   └── "Did the installation work?"
│
├── 📄 INDEX.md (this file) ............... NAVIGATION
│   └── "Where should I read?"
│
└── 📄 visualize_training.py .............. UTILITY
    └── python training/visualize_training.py --help
```

---

## Suggested Reading Sessions

### Session 1: Get Oriented (10 min)
1. Read this INDEX.md (2 min)
2. Read README_ACCURACY.md (5 min)
3. Run your first training (3 min)

### Session 2: Learn to Interpret (15 min)
1. Read ACCURACY_QUICK_REF.md (5 min)
2. Run visualize_training.py (3 min)
3. Study interpretation table (7 min)

### Session 3: Deep Understanding (20 min)
1. Read ACCURACY_METRICS_README.md (15 min)
2. Read IMPLEMENTATION_SUMMARY.md (5 min)

### Session 4: Troubleshoot (10 min)
1. Read CHECKLIST.md (5 min)
2. Verify files modified (5 min)

---

**Total Reading Time: 45-55 minutes for complete understanding**  
**Essential Reading: 10 minutes to get started**

---

Pick a document above and dive in! 🚀

Last Updated: January 22, 2026
