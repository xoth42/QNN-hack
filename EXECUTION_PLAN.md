# Quantum CNN Comparison - Complete Execution Plan

## 🎯 Project Goal
Compare classical CNN vs quantum-hybrid CNN on CIFAR-10 and determine if quantum layers provide any advantage.

---

## 📋 Phase 1: Setup & Verification (DO THIS FIRST)

### Task 1.1: Configure Git Identity
**What**: Set up your git credentials so you can commit changes  
**Why**: Required before making any commits  
**How**:
```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```
**Verify**: Run `git config --list` and check your name/email appear  
**Status**: ⏸️ NOT STARTED

---

### Task 1.2: Setup Python Environment
**What**: Create conda environment with all dependencies  
**Why**: Need PyTorch, torchvision, matplotlib, numpy installed  
**How**:
```bash
cd QNN-hack
bash setup_cnn_env.sh
conda activate cnn
```
**Verify**: Run `python -c "import torch; print(torch.__version__)"` - should print version  
**Status**: ⏸️ NOT STARTED

---

### Task 1.3: Test Classical CNN (Baseline Check)
**What**: Run the existing classical CNN to verify it works  
**Why**: Make sure baseline code is functional before proceeding  
**How**:
```bash
conda activate cnn
python cifar10_tinycnn.py
```
**Expected**: Should download CIFAR-10, show sample images, start training  
**Time**: ~15-20 minutes for 15 epochs  
**Verify**: Check for `best_model.pth` file created and final test accuracy printed  
**Status**: ⏸️ NOT STARTED

---

### Task 1.4: Install Quantum Dependencies
**What**: Install PennyLane and AWS Braket SDK  
**Why**: Required for quantum layer functionality  
**How**:
```bash
conda activate cnn
pip install pennylane pennylane-braket amazon-braket-sdk
```
**Verify**: Run `python -c "import pennylane as qml; print(qml.__version__)"`  
**Status**: ⏸️ NOT STARTED

---

### Task 1.5: Create Verification Script
**What**: Build a script that tests all components without running full training  
**Why**: Quick sanity check before expensive experiments  
**File**: `verify_setup.py`  
**Tests**:
- Import all libraries
- Load small batch of CIFAR-10
- Forward pass through Classical_CNN
- Forward pass through Quantum_Hybrid_CNN (local simulator)
- Test Experiment_Tracker save/load

**How**: I'll create this file for you  
**Verify**: Run `python verify_setup.py` - all tests should pass  
**Status**: ⏸️ NOT STARTED

---

## 📊 Phase 2: Classical Baseline Experiments

### Task 2.1: Run Classical Baseline (Full)
**What**: Complete training run of classical CNN with tracking  
**Why**: Establish performance benchmark  
**How**:
```bash
python cifar10_tinycnn.py
```
**Expected Output**:
- Training/validation loss per epoch
- Final test accuracy (~70-75%)
- Training time (~15-20 min)
- Saved model: `best_model.pth`

**Verify**: Check that training completes and accuracy is reasonable  
**Status**: ⏸️ NOT STARTED

---

### Task 2.2: Add Tracking to Classical CNN
**What**: Integrate ExperimentTracker into cifar10_tinycnn.py  
**Why**: Need consistent metrics for comparison  
**Changes**:
- Import track_performance
- Create tracker at start
- Log epochs during training
- Save results at end

**How**: I'll modify the file for you  
**Verify**: Check `experiments/classical/` folder has JSON file with metrics  
**Status**: ⏸️ NOT STARTED

---

### Task 2.3: Generate Classical Visualizations
**What**: Create plots and analysis of classical baseline  
**Why**: Need presentable results for comparison  
**Outputs**:
- Loss curves (train/val)
- Accuracy over time
- Confusion matrix
- Summary statistics

**How**: I'll create `visualize_results.py` script  
**Verify**: Check that plots are generated and look correct  
**Status**: ⏸️ NOT STARTED

---

## 🔬 Phase 3: Quantum Hybrid Experiments

### Task 3.1: Test Quantum Layer (Local Simulator)
**What**: Run quantum hybrid CNN with local simulator (no AWS)  
**Why**: Test functionality without AWS costs  
**How**:
```bash
# First, I'll modify quantum_hybrid_cnn.py to add --local flag
python quantum_hybrid_cnn.py --local --epochs 2 --batch-size 16
```
**Expected**: Should complete 2 epochs quickly on CPU  
**Verify**: No errors, forward/backward pass works  
**Status**: ⏸️ NOT STARTED

---

### Task 3.2: Configure AWS Braket (If Available)
**What**: Set up AWS credentials for quantum simulator  
**Why**: Required to run on AWS quantum simulator  
**How**:
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1)
```
**Verify**: Run `aws sts get-caller-identity` - should show your account  
**Status**: ⏸️ NOT STARTED (OPTIONAL - can use local simulator)

---

### Task 3.3: Run Quantum Hybrid (4 Qubits, 2 Layers)
**What**: First quantum experiment with standard configuration  
**Why**: Baseline quantum performance  
**How**:
```bash
python quantum_hybrid_cnn.py --epochs 10 --batch-size 16 --quantum-qubits 4 --quantum-layers 2
```
**Expected**:
- Slower than classical (quantum overhead)
- Accuracy: unknown (this is the experiment!)
- Auto-saved to `experiments/quantum/`

**Verify**: Training completes, metrics saved  
**Status**: ⏸️ NOT STARTED

---

### Task 3.4: Experiment - Vary Qubit Count
**What**: Test with 2, 4, 8 qubits  
**Why**: See how quantum dimension affects performance  
**How**:
```bash
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 2
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 8
```
**Expected**: Different accuracy/time tradeoffs  
**Verify**: 3 JSON files in `experiments/quantum/`  
**Status**: ⏸️ NOT STARTED

---

### Task 3.5: Experiment - Vary Circuit Depth
**What**: Test with 1, 2, 3 quantum layers  
**Why**: See if deeper circuits help  
**How**:
```bash
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4 --quantum-layers 1
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4 --quantum-layers 2
python quantum_hybrid_cnn.py --epochs 10 --quantum-qubits 4 --quantum-layers 3
```
**Expected**: Deeper = slower but possibly better accuracy  
**Verify**: 3 JSON files with different configurations  
**Status**: ⏸️ NOT STARTED

---

## 📈 Phase 4: Analysis & Comparison

### Task 4.1: Create Comparison Script
**What**: Build tool to compare all experiments  
**Why**: Need automated analysis of results  
**File**: `compare_results.py`  
**Features**:
- Load all experiments from JSON files
- Calculate averages and std dev
- Generate comparison plots
- Create summary table

**How**: I'll create this script  
**Verify**: Run `python compare_results.py` and check output  
**Status**: ⏸️ NOT STARTED

---

### Task 4.2: Generate Final Comparison Report
**What**: Run comparison on all completed experiments  
**Why**: Final deliverable for presentation  
**How**:
```bash
python compare_results.py --output comparison_report.pdf
```
**Expected Output**:
- Table: Classical vs Quantum accuracy
- Plot: Training time comparison
- Plot: Accuracy vs qubit count
- Plot: Accuracy vs circuit depth
- Summary findings

**Verify**: Report looks professional and complete  
**Status**: ⏸️ NOT STARTED

---

### Task 4.3: Document Findings
**What**: Write up results and conclusions  
**Why**: Explain what you learned  
**File**: `RESULTS.md`  
**Sections**:
- Executive summary
- Classical baseline results
- Quantum hybrid results
- Key findings
- Limitations
- Future work

**How**: I'll create template, you fill in results  
**Verify**: Document is clear and complete  
**Status**: ⏸️ NOT STARTED

---

## 🚀 Phase 5: Finalization & Submission

### Task 5.1: Code Cleanup
**What**: Clean up code, add comments, remove debug prints  
**Why**: Professional submission  
**Files to review**:
- All .py files
- Remove any hardcoded paths
- Ensure consistent style

**Verify**: Code review checklist  
**Status**: ⏸️ NOT STARTED

---

### Task 5.2: Update Documentation
**What**: Ensure all README files are accurate  
**Why**: Others need to understand your work  
**Files**:
- README.md (main project description)
- QUICKSTART.md (verify instructions work)
- CONTRIBUTING.md (update if needed)

**Verify**: Fresh team member could follow docs  
**Status**: ⏸️ NOT STARTED

---

### Task 5.3: Commit All Changes
**What**: Commit everything to git  
**Why**: Version control and submission  
**How**:
```bash
git add .
git commit -m "Complete quantum CNN comparison project"
git push origin quantum-layer-development
```
**Verify**: Check GitHub - all files uploaded  
**Status**: ⏸️ NOT STARTED

---

### Task 5.4: Create Pull Request
**What**: Submit PR from quantum-layer-development to main  
**Why**: Official submission  
**How**:
- Go to GitHub repository
- Click "New Pull Request"
- Select: base=main, compare=quantum-layer-development
- Write description of work
- Submit

**Verify**: PR created and visible  
**Status**: ⏸️ NOT STARTED

---

## 📝 Quick Reference Commands

### Daily Workflow
```bash
# Start working
cd QNN-hack
conda activate cnn
git status

# Run experiments
python cifar10_tinycnn.py                    # Classical
python quantum_hybrid_cnn.py --epochs 10     # Quantum

# Check results
ls experiments/classical/
ls experiments/quantum/

# Commit progress
git add .
git commit -m "Description of what you did"
git push origin quantum-layer-development
```

---

## ⚠️ Important Notes

1. **Start with Phase 1** - Don't skip verification steps
2. **One task at a time** - Complete and verify before moving on
3. **Save everything** - All experiments auto-save to experiments/
4. **AWS costs** - Use local simulator for testing, AWS only for final runs
5. **Time estimates**:
   - Classical training: ~15-20 min
   - Quantum training: ~30-60 min (depends on AWS/local)
   - Total project: ~4-6 hours of compute time

---

## 🆘 Troubleshooting

**Problem**: Import errors  
**Solution**: Make sure conda environment is activated: `conda activate cnn`

**Problem**: CUDA out of memory  
**Solution**: Reduce batch size: `--batch-size 16` or `--batch-size 8`

**Problem**: Quantum layer errors  
**Solution**: Start with local simulator: `--local` flag

**Problem**: AWS Braket access denied  
**Solution**: Use local simulator or check AWS credentials

---

## ✅ Success Criteria

You're done when:
- [ ] Classical baseline runs and achieves ~70-75% accuracy
- [ ] Quantum hybrid runs without errors
- [ ] At least 3 quantum experiments completed (different configs)
- [ ] Comparison report generated
- [ ] All code committed and PR submitted
- [ ] Results documented

---

## 📞 Next Steps

**Tell me which task you want to start with**, and I'll:
1. Create any missing files needed for that task
2. Walk you through the exact commands
3. Help verify it worked correctly
4. Move to the next task

**Recommended starting point**: Task 1.2 (Setup Python Environment)
