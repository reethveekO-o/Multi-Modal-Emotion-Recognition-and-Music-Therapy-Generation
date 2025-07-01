EMOTION_MAP = {
    ('anger','high'):'Rage', ('anger','mid'):'Anger', ('anger','low'):'Annoyance',
    ('sadness','high'):'Despair', ('sadness','mid'):'Sadness', ('sadness','low'):'Melancholy',
    ('joy','high'):'Excitement', ('joy','mid'):'Happiness', ('joy','low'):'Contentment',
    ('fear','high'):'Panic', ('fear','mid'):'Fear', ('fear','low'):'Anxiety',
    ('love','high'):'Passion', ('love','mid'):'Love', ('love','low'):'Warmth',
    ('surprise','high'):'Shock', ('surprise','mid'):'Surprise', ('surprise','low'):'Curiosity'
}

def build_prompt(emotion:str, stress:str)->str:
    val = EMOTION_MAP[(emotion, stress)]
    tempo = {'high':'60','mid':'70','low':'80'}[stress]
    freq  = {'high':'432','mid':'528','low':'640'}[stress]
    mood  = {
       'Rage':'steady, comforting resolve',
       'Despair':'tender uplifting hope',
       'Panic':'soft grounding safety',
       'Anxiety':'calming warm reassurance',
       'Sadness':'gentle brightness',
       'Melancholy':'subtle optimism',
       'Anger':'soothing balance',
       'Annoyance':'light curiosity',
       'Excitement':'sustained joyful energy',
       'Happiness':'peaceful happiness',
       'Contentment':'quiet contented glow',
       'Passion':'warm affectionate atmosphere',
       'Love':'embracing warmth',
       'Warmth':'cozy serenity',
       'Shock':'gradual wonder',
       'Surprise':'pleasant intrigue',
       'Curiosity':'soft playful exploration'
    }[val]
    return (f"{mood} instrumental, {tempo} BPM, centered around {freq} Hz drone, "
            "major mode with soft pads, minimal percussion, long reverb, "
            "composed to reduce stress and instill positivity")
# Example usage
prompt = build_prompt('anger','high')
print(prompt)
