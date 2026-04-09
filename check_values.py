from env.email_env import EmailManagementEnv
from env.models import Action, ActionType, EmailLabel, PriorityLevel

tasks = [
    ('spam_detection',       Action(action_type=ActionType.CLASSIFY,   label=EmailLabel('spam'))),
    ('email_prioritization', Action(action_type=ActionType.PRIORITIZE, priority=PriorityLevel('high'))),
    ('auto_reply',           Action(action_type=ActionType.REPLY,       reply_text='no_reply')),
]

print(f"{'Task':<25} {'Step':>4}  {'reward.value':>14}  {'score':>8}  {'mean_score':>12}  {'breakdown'}")
print("-" * 90)

all_ok = True
for task, action in tasks:
    env = EmailManagementEnv(task_name=task)
    obs = env.reset()
    step = 0
    while obs is not None:
        step += 1
        obs, reward, done, info = env.step(action)
        rv  = reward.value
        sc  = info['score']
        ms  = info['mean_score']
        bd  = reward.breakdown
        ok  = (0 < rv < 1) and (0 < sc < 1) and (0 < ms < 1)
        all_bad = [f"{k}={v}" for k, v in bd.items() if not (0 < v < 1)]
        status = "OK" if (ok and not all_bad) else "FAIL!!!"
        print(f"{task:<25} {step:>4}  {rv:>14.6f}  {sc:>8.6f}  {ms:>12.6f}  {bd}  [{status}]")
        if not ok or all_bad:
            all_ok = False

print("-" * 90)
print("ALL VALUES STRICTLY BETWEEN 0 AND 1 ✓" if all_ok else "FAILURES DETECTED ✗")
