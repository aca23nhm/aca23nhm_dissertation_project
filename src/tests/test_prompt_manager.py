from src.prompt_manager import load_prompts, render_prompt

templates = load_prompts("prompts")
s = "She go to university every day."

for pid, tmpl in templates.items():
    print("=" * 60)
    print(pid)
    print(render_prompt(tmpl, s))