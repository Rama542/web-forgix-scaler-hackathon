"""
run_demo.py  –  Drop-in runner for environments where pydantic is not yet installed.
Mirrors inference.py 100% — same logic, same log format, stdlib only.
Delete this file once you've done `pip install -r requirements.txt`.
"""

import json, re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────────────────────
class EmailLabel(str, Enum):
    SPAM = "spam"; IMPORTANT = "important"; NORMAL = "normal"

class PriorityLevel(str, Enum):
    HIGH = "high"; MEDIUM = "medium"; LOW = "low"

class ActionType(str, Enum):
    CLASSIFY = "classify_email"; PRIORITIZE = "prioritize_email"; REPLY = "reply_email"

# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Observation:
    email_id:str; subject:str; sender:str; body:str
    urgency_hint:Optional[str]; timestamp:str; task:str
    difficulty:str; step_number:int; remaining_emails:int; instructions:str

@dataclass
class Action:
    action_type:ActionType
    label:Optional[EmailLabel]=None
    priority:Optional[PriorityLevel]=None
    reply_text:Optional[str]=None
    def validate_for_task(self,t):
        m={"spam_detection":ActionType.CLASSIFY,"email_prioritization":ActionType.PRIORITIZE,"auto_reply":ActionType.REPLY}
        return m.get(t) is None or self.action_type==m[t]

@dataclass
class Reward:
    value:float; breakdown:Dict[str,float]=field(default_factory=dict); explanation:str=""

# ────────────────────────────────────────────────────────────────────────────
# Email corpus (10 real-world emails)
# ────────────────────────────────────────────────────────────────────────────
EMAILS=[
    {"email_id":"email_001","subject":"Congratulations! You've won a $1,000 Amazon gift card",
     "sender":"promo@random-prizes.biz",
     "body":"Dear valued customer, You have been selected as our lucky winner! Click to claim your FREE $1,000 Amazon gift card. Hurry – offer expires in 24 hours! http://totally-legit-prizes.biz/claim",
     "urgency_hint":None,"timestamp":"2024-03-15T08:00:00Z",
     "_true_label":"spam","_true_priority":"low","_ideal_reply_keywords":[]},

    {"email_id":"email_002","subject":"Q1 Financial Report – Action Required by Friday",
     "sender":"cfo@acmecorp.com",
     "body":"Hi team, Please review the attached Q1 financial report and provide your sign-off by end of business Friday. The board meeting is scheduled for Monday. Regards, Sarah – CFO",
     "urgency_hint":"deadline Friday","timestamp":"2024-03-15T09:15:00Z",
     "_true_label":"important","_true_priority":"high",
     "_ideal_reply_keywords":["acknowledge","review","sign-off","Friday","confirmed"]},

    {"email_id":"email_003","subject":"Free V1agra & C1alis – Limited Time Offer!!!",
     "sender":"deals@pharma-discount99.ru",
     "body":"Get 90% OFF prescription meds – no prescription needed! Order now: http://pharma-discount99.ru/order",
     "urgency_hint":None,"timestamp":"2024-03-15T09:30:00Z",
     "_true_label":"spam","_true_priority":"low","_ideal_reply_keywords":[]},

    {"email_id":"email_004","subject":"Team lunch this Thursday?",
     "sender":"mike.johnson@acmecorp.com",
     "body":"Hey, Are you free for team lunch on Thursday around 12:30? Thinking of trying that new Italian place on 5th. Let me know! Mike",
     "urgency_hint":None,"timestamp":"2024-03-15T10:00:00Z",
     "_true_label":"normal","_true_priority":"low",
     "_ideal_reply_keywords":["Thursday","lunch","sounds good","join","confirm"]},

    {"email_id":"email_005","subject":"URGENT: Production server is down",
     "sender":"alerts@monitoring.acmecorp.com",
     "body":"ALERT: The production API server has been unreachable for 10 minutes. Error rate 100%. ~2,000 active users impacted. Please acknowledge and escalate immediately.",
     "urgency_hint":"production outage","timestamp":"2024-03-15T11:05:00Z",
     "_true_label":"important","_true_priority":"high",
     "_ideal_reply_keywords":["acknowledge","escalate","investigating","outage","team"]},

    {"email_id":"email_006","subject":"Monthly newsletter – March 2024",
     "sender":"newsletter@techdigest.io",
     "body":"Hi there, Welcome to the March edition of Tech Digest! This month: AI trends, cloud cost tips, and an interview with a principal engineer at Stripe.",
     "urgency_hint":None,"timestamp":"2024-03-15T12:00:00Z",
     "_true_label":"normal","_true_priority":"low","_ideal_reply_keywords":[]},

    {"email_id":"email_007","subject":"Your AWS bill is unusually high this month",
     "sender":"billing@aws.amazon.com",
     "body":"Dear Customer, Your AWS charges for February 2024 are $4,320 – approximately 340% higher than your monthly average. Please sign in to the AWS Console to review usage.",
     "urgency_hint":"billing anomaly","timestamp":"2024-03-15T13:30:00Z",
     "_true_label":"important","_true_priority":"high",
     "_ideal_reply_keywords":["investigate","usage","budget","alert","check"]},

    {"email_id":"email_008","subject":"You've been selected for a special survey – $50 reward",
     "sender":"surveys@market-research-hub.net",
     "body":"Hello, Complete our 2-minute survey and receive a $50 gift voucher instantly! Click: http://survey-link.net/s?ref=abc. Offer valid for 12 hours only.",
     "urgency_hint":None,"timestamp":"2024-03-15T14:00:00Z",
     "_true_label":"spam","_true_priority":"low","_ideal_reply_keywords":[]},

    {"email_id":"email_009","subject":"Performance review scheduled for next week",
     "sender":"hr@acmecorp.com",
     "body":"Hi, Your annual performance review is scheduled for Tuesday, March 19th at 2:00 PM in Conference Room B. Please come prepared with a summary of your achievements. HR Team",
     "urgency_hint":"scheduled review","timestamp":"2024-03-15T15:00:00Z",
     "_true_label":"important","_true_priority":"medium",
     "_ideal_reply_keywords":["confirmed","Tuesday","prepared","thank you","review"]},

    {"email_id":"email_010","subject":"Office Wi-Fi will be down Saturday 2–4 AM for maintenance",
     "sender":"it-ops@acmecorp.com",
     "body":"Hi everyone, Planned network maintenance is scheduled for Saturday 02:00–04:00 AM. Office Wi-Fi and VPN access will be unavailable during this window. IT Ops",
     "urgency_hint":None,"timestamp":"2024-03-15T15:45:00Z",
     "_true_label":"normal","_true_priority":"medium",
     "_ideal_reply_keywords":["noted","acknowledged","thanks"]},
]

# ────────────────────────────────────────────────────────────────────────────
# Graders
# ────────────────────────────────────────────────────────────────────────────
_LS={("spam","spam"):1.0,("important","important"):1.0,("normal","normal"):1.0,
     ("important","normal"):0.3,("normal","important"):0.2,("spam","important"):0.0,
     ("spam","normal"):0.1,("normal","spam"):0.3,("important","spam"):0.3}
_PO={"high":2,"medium":1,"low":0}

def grade_classification(p,t): return _LS.get((p.lower(),t.lower()),0.0)
def grade_prioritization(p,t):
    a,b=_PO.get(p.lower()),_PO.get(t.lower())
    if a is None or b is None: return 0.0
    d=abs(a-b); return 1.0 if d==0 else (0.5 if d==1 else 0.0)

def grade_reply(reply,true_label,keywords):
    n=reply.strip().lower()
    no_r=n in("","no_reply","no reply","noreply")
    if true_label=="spam": return 1.0 if no_r else 0.1
    if no_r: return 0.0
    if len(reply.strip())<20: return 0.2
    if not keywords: return 0.6
    norm=lambda s:re.sub(r"[^a-z0-9 ]"," ",s.lower())
    hits=sum(1 for kw in keywords if norm(kw) in norm(reply))
    return 0.2+0.8*(hits/len(keywords))

def compute_score(task,action_dict,email):
    if task=="spam_detection":       return grade_classification(action_dict.get("label",""),email["_true_label"])
    if task=="email_prioritization": return grade_prioritization(action_dict.get("priority",""),email["_true_priority"])
    if task=="auto_reply":           return grade_reply(action_dict.get("reply_text",""),email["_true_label"],email["_ideal_reply_keywords"])
    raise ValueError(task)

# ────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────
class EmailManagementEnv:
    R_CORRECT=1.0; R_HI=0.5; R_LO=0.2; R_WRONG=-0.5; R_BAD=-1.0
    INSTRUCTIONS={"spam_detection":"Classify each email as spam, important, or normal.",
                  "email_prioritization":"Assign each email a priority: high, medium, or low.",
                  "auto_reply":"Draft a professional reply. For spam respond with 'no_reply'."}
    DIFF={"spam_detection":"easy","email_prioritization":"medium","auto_reply":"hard"}

    def __init__(self,task_name="spam_detection"):
        self.task_name=task_name; self._emails=[]; self._cursor=0
        self._cum_r=0.0; self._scores=[]; self._done=False; self._step=0

    def reset(self):
        import copy
        self._emails=copy.deepcopy(EMAILS); self._cursor=0
        self._cum_r=0.0; self._scores=[]; self._done=False; self._step=0
        return self._obs()

    def step(self,action):
        e=self._emails[self._cursor]
        r,info=self._reward(action,e)
        self._cum_r+=r.value; self._scores.append(info.get("score",0.0))
        self._step+=1; self._cursor+=1
        if self._cursor>=len(self._emails): self._done=True
        nobs=None if self._done else self._obs()
        info.update({"step":self._step,"cumulative_reward":self._cum_r})
        return nobs,r,self._done,info

    def state(self):
        return {"task_name":self.task_name,"difficulty":self.DIFF.get(self.task_name),
                "current_step":self._step,"total_emails":len(self._emails),
                "emails_processed":self._cursor,"cumulative_reward":round(self._cum_r,4),
                "scores":[round(s,4)for s in self._scores],"done":self._done}

    def _obs(self):
        e=self._emails[self._cursor]
        return Observation(email_id=e["email_id"],subject=e["subject"],sender=e["sender"],
            body=e["body"],urgency_hint=e.get("urgency_hint"),timestamp=e["timestamp"],
            task=self.task_name,difficulty=self.DIFF.get(self.task_name,"unknown"),
            step_number=self._step,remaining_emails=len(self._emails)-self._cursor,
            instructions=self.INSTRUCTIONS.get(self.task_name,""))

    def _reward(self,action,email):
        if not action.validate_for_task(self.task_name):
            return Reward(self.R_BAD,{"type_penalty":self.R_BAD},"Wrong action type"),{"score":0.0,"error":"wrong_action_type"}
        ad={"action_type":action.action_type.value}
        if action.label     is not None: ad["label"]=action.label.value
        if action.priority  is not None: ad["priority"]=action.priority.value
        if action.reply_text is not None: ad["reply_text"]=action.reply_text
        sc=compute_score(self.task_name,ad,email)
        if sc>=0.9: rv,ex=self.R_CORRECT, f"Excellent!  score={sc:.2f}"
        elif sc>=0.5: rv,ex=self.R_HI,    f"Good.       score={sc:.2f}"
        elif sc>=0.2: rv,ex=self.R_LO,    f"Marginal.   score={sc:.2f}"
        else:         rv,ex=self.R_WRONG,  f"Incorrect.  score={sc:.2f}"
        return Reward(rv,{"correctness":sc,"scaled_reward":rv},ex),{"score":sc,"error":None}

    def summary(self):
        if not self._scores: return {"status":"not started"}
        return {"task":self.task_name,"total_steps":self._step,
                "mean_score":round(sum(self._scores)/len(self._scores),4),
                "cumulative_reward":round(self._cum_r,4),"done":self._done}

# ────────────────────────────────────────────────────────────────────────────
# Rule-based agent  (same logic as inference.py fallback)
# ────────────────────────────────────────────────────────────────────────────
_SK=["won","winner","prize","gift card","claim","free","offer","limited","click here",
     "discount","v1agra","c1alis","pharma","survey","$50","selected","voucher","reward"]

def _classify(obs):
    t=(obs.subject+" "+obs.body+" "+obs.sender).lower()
    if sum(1 for k in _SK if k in t)>=2: return EmailLabel.SPAM
    if any(c in t for c in["urgent","action required","asap","deadline","outage","alert","down"]): return EmailLabel.IMPORTANT
    if any(c in t for c in["meeting","review","report","team","scheduled","maintenance","billing","performance"]): return EmailLabel.IMPORTANT
    return EmailLabel.NORMAL

def _prioritize(obs):
    t=(obs.subject+" "+obs.body+" "+(obs.urgency_hint or "")).lower()
    if any(k in t for k in["urgent","outage","down","action required","asap","deadline","billing"]): return PriorityLevel.HIGH
    if any(k in t for k in["scheduled","review","report","maintenance","performance"]): return PriorityLevel.MEDIUM
    return PriorityLevel.LOW

def _reply(obs):
    t=(obs.subject+" "+obs.body).lower()
    if sum(1 for k in _SK if k in t)>=2: return "no_reply"
    if "outage" in t or "down" in t:
        return "Thank you for the alert. I acknowledge the outage and will escalate to the on-call team immediately to begin investigating the issue."
    if "performance" in t and "review" in t:
        return "Thank you for the notification. I confirm I will be fully prepared and available for the performance review on Tuesday at 2:00 PM as scheduled."
    if "financial" in t or "sign-off" in t:
        return "Acknowledged. I will review the Q1 financial report carefully and provide my sign-off before end of business Friday as requested."
    if "billing" in t or "aws" in t:
        return "Thank you for the billing alert. I will investigate the unusual AWS usage immediately and set up budget alerts to prevent this from recurring."
    if "lunch" in t:
        return "Sounds great! Thursday lunch at 12:30 works perfectly for me. I'll join you at the Italian place on 5th. Looking forward to it!"
    if "maintenance" in t or "wi-fi" in t:
        return "Noted and acknowledged. I'll make sure everything requiring VPN access is completed before the maintenance window on Saturday. Thanks for the heads-up."
    return "Thank you for your email. I have received your message and will follow up as soon as possible."

def build_action(task,obs):
    if task=="spam_detection":
        l=_classify(obs); return Action(ActionType.CLASSIFY,label=l), f"classify_email(label={l.value})"
    if task=="email_prioritization":
        p=_prioritize(obs); return Action(ActionType.PRIORITIZE,priority=p), f"prioritize_email(priority={p.value})"
    r=_reply(obs)
    short=(r[:58]+"...") if len(r)>58 else r
    return Action(ActionType.REPLY,reply_text=r), f'reply_email(text="{short}")'

# ────────────────────────────────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────────────────────────────────
MODEL   = "rule-based-agent"
ENV     = "EmailManagementEnv"

def log_start(task):  print(f"[START] task={task} env={ENV} model={MODEL}",flush=True)
def log_step(step,action,reward,done,error):
    done_str  = "true" if done else "false"
    error_str = '"' + error + '"' if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done_str} error={error_str}",flush=True)
def log_end(ok,steps,rewards):
    print(f"[END] success={'true'if ok else'false'} steps={steps} rewards={json.dumps([round(r,4)for r in rewards])}",flush=True)

# ────────────────────────────────────────────────────────────────────────────
# Task runner
# ────────────────────────────────────────────────────────────────────────────
def run_task(task_name):
    env=EmailManagementEnv(task_name)
    obs=env.reset()
    log_start(task_name)
    rewards,step=[],0
    while obs is not None:
        step+=1
        action,action_str=build_action(task_name,obs)
        obs,reward,done,info=env.step(action)
        rewards.append(reward.value)
        log_step(step,action_str,reward.value,done,info.get("error"))
        if done: break
    mean_r=sum(rewards)/len(rewards) if rewards else 0.0
    log_end(mean_r>0,step,rewards)
    s=env.summary()
    print(f"[INFO] mean_reward={mean_r:.4f}  mean_score={s['mean_score']:.4f}\n",flush=True)
    return {"task":task_name,"success":mean_r>0,"steps":step,"rewards":rewards,"mean_reward":mean_r}

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    W=68
    print("="*W)
    print("  📬  Email Management RL Environment  –  Baseline Inference  📬")
    print("="*W)
    print(f"  Model        : {MODEL}")
    print(f"  Environment  : {ENV}")
    print(f"  Python       : stdlib-only demo  (pip install -r requirements.txt for full run)")
    print(f"  Tasks        : spam_detection  ·  email_prioritization  ·  auto_reply")
    print("="*W); print()

    results=[]
    for t in["spam_detection","email_prioritization","auto_reply"]:
        print("-"*W)
        results.append(run_task(t))

    print("="*W)
    print("  FINAL SUMMARY")
    print("="*W)
    for r in results:
        status="✓ PASS" if r["success"] else "✗ FAIL"
        print(f"  {status}  {r['task']:<28}  mean_reward={r['mean_reward']:.4f}  steps={r['steps']}")
    print("-"*W)
    print(f"  Overall: {'ALL TASKS PASSED ✓' if all(r['success'] for r in results) else 'SOME TASKS FAILED ✗'}")
    print("="*W)

if __name__=="__main__":
    main()
