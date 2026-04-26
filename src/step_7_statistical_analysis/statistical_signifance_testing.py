import scipy.stats as stats

baseline = [59.37, 60.83, 58.75]
instruction = [66.05, 66.92, 65.39]
role = [63.32, 64.31, 62.55]
fewshot = [67.53, 68.61, 66.79]

def test(a, b, name):
    stat, p = stats.wilcoxon(a, b)
    print(name, "p =", round(p, 4))

# F0.5 comparisons
test(instruction, baseline, "Instruction vs Baseline")
test(role, baseline, "Role vs Baseline")
test(fewshot, baseline, "Few-shot vs Baseline")
test(instruction, role, "Instruction vs Role")
test(fewshot, role, "Few-shot vs Role")
test(fewshot, instruction, "Few-shot vs Instruction")