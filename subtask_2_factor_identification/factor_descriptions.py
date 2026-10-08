"""
One-sentence clinical descriptions per factor, in the SAME order as
config.FACTOR_LIST. Used by the dual-encoder model (v14 / backlog item:
label-semantic embeddings) as the text input for the "factor encoder" side —
giving rare factors (only 5-20 positive training examples for some) a
semantic anchor beyond what a randomly-initialized classification weight
vector could learn from so few examples alone.
"""

from config import FACTOR_LIST

FACTOR_DESCRIPTIONS_BY_NAME = {
    "mental health issues": "The person mentions a diagnosed or suspected mental health condition such as depression, anxiety, bipolar disorder, or PTSD.",
    "physical health/characteristic": "The person mentions a physical illness, disability, chronic pain, or a physical characteristic causing them distress.",
    "substance use": "The person mentions using or abusing alcohol, drugs, or other substances.",
    "hopelessness": "The person expresses a sense that things will never get better, that there is no way out, or that the future is bleak.",
    "emotion dysregulation": "The person describes intense, overwhelming, or uncontrollable emotions such as rage, panic, or emotional numbness.",
    "low self-esteem": "The person expresses feeling worthless, inadequate, unlovable, or like a failure.",
    "poor school performance": "The person mentions failing grades, academic struggles, or difficulty keeping up at school.",
    "low socio-economic status": "The person mentions poverty, financial hardship, unemployment, or inability to afford basic needs.",
    "interpersonal violence": "The person mentions being a victim of physical, sexual, or emotional abuse or violence from another person.",
    "prior self-harm or suicidal thought/attempt": "The person mentions a previous instance of self-harm, suicidal ideation, or a suicide attempt.",
    "poor social support": "The person expresses feeling alone, having no one to talk to, or lacking friends or family support.",
    "interpersonal difficulty": "The person describes conflict, arguments, or a breakdown in a relationship with a friend, partner, or family member.",
    "dysfunctional family": "The person describes a troubled, abusive, neglectful, or chaotic family environment.",
    "exposure to others' suicide": "The person mentions knowing someone, such as a friend or family member, who died by suicide or attempted suicide.",
    "stressful life event": "The person mentions a recent significant negative life event such as a breakup, job loss, or death of a loved one.",
    "traumatic experience": "The person mentions a past traumatic experience such as an accident, assault, or disaster.",
    "cognitive deficits": "The person mentions difficulty concentrating, memory problems, confusion, or impaired decision-making.",
    "suicide means (with access)": "The person mentions having access to a specific method of suicide, such as pills, a weapon, or a high place.",
    "sexual orientation related issues": "The person mentions distress related to their sexual orientation, such as rejection, discrimination, or difficulty accepting their identity.",
    "social support": "The person mentions having friends, family, or others who care about them and provide emotional support.",
    "coping strategy": "The person mentions a specific strategy they use to manage distress, such as exercise, therapy, journaling, or talking to someone.",
    "psychological capital": "The person expresses resilience, optimism, self-efficacy, or confidence in their ability to overcome difficulties.",
    "sense of responsibility": "The person mentions feeling responsible for someone else, such as children, family, or pets, as a reason to keep going.",
    "meaning in life": "The person expresses having a sense of purpose, goals, or something meaningful to live for.",
}

# Ordered to exactly match config.FACTOR_LIST — this order is what every
# script in this file iterates over.
FACTOR_DESCRIPTIONS = [FACTOR_DESCRIPTIONS_BY_NAME[f] for f in FACTOR_LIST]

assert len(FACTOR_DESCRIPTIONS) == len(FACTOR_LIST), "Missing a factor description"
