from stress_final import main
stress_score=main()
emotion='sadness'
def generate_music_prompt(stress_score, emotion):
    # Classify stress level
    if stress_score >= 0.66:
        stress_level = 'high'
        freq = '432'
    elif stress_score >= 0.33:
        stress_level = 'mid'
        freq = '528'
    else:
        stress_level = 'low'
        freq = '640'

    # Emotion category mapping
    emotion_categories = {
        'rage': {'type': 'high_negative', 'synonyms': ['anger', 'irritation']},
        'anger': {'type': 'high_negative', 'synonyms': ['rage', 'annoyance', 'irritation']},
        'annoyance': {'type': 'mid_negative', 'synonyms': ['irritation', 'anger']},
        'irritation': {'type': 'mid_negative', 'synonyms': ['annoyance', 'anger']},
        'despair': {'type': 'low_negative', 'synonyms': ['sadness', 'melancholy']},
        'sadness': {'type': 'low_negative', 'synonyms': ['despair', 'melancholy']},
        'melancholy': {'type': 'low_negative', 'synonyms': ['sadness', 'despair']},
        'excitement': {'type': 'high_positive', 'synonyms': ['happiness']},
        'happiness': {'type': 'high_positive', 'synonyms': ['excitement', 'contentment']},
        'contentment': {'type': 'low_positive', 'synonyms': ['happiness']},
        'panic': {'type': 'high_fear', 'synonyms': ['fear', 'anxiety']},
        'fear': {'type': 'mid_fear', 'synonyms': ['panic', 'anxiety']},
        'anxiety': {'type': 'mid_fear', 'synonyms': ['fear', 'panic']},
        'passion': {'type': 'high_love', 'synonyms': ['love', 'warmth']},
        'love': {'type': 'mid_love', 'synonyms': ['passion', 'warmth']},
        'warmth': {'type': 'low_love', 'synonyms': ['love', 'passion']},
        'shock': {'type': 'high_surprise', 'synonyms': ['surprise']},
        'surprise': {'type': 'mid_surprise', 'synonyms': ['shock', 'curiosity']},
        'curiosity': {'type': 'low_surprise', 'synonyms': ['surprise']}
    }

    base_emotion = next((emo for emo in emotion_categories if emotion.lower() == emo or emotion.lower() in emotion_categories[emo]['synonyms']), 'contentment')
    emotion_type = emotion_categories[base_emotion]['type']

    therapy_static = {
        'high_negative':   'heavy and grounded with slow motion',
        'mid_negative':    'gentle and flowing',
        'low_negative':    'soft and emotionally warm',
        'high_positive':   'bright and rhythmic',
        'low_positive':    'peaceful and steady',
        'high_fear':       'deep and steady',
        'mid_fear':        'calm and predictable',
        'high_love':       'rich and emotional',
        'mid_love':        'soft and melodic',
        'low_love':        'light and soothing',
        'high_surprise':   'shifting but cohesive',
        'mid_surprise':    'quirky and interesting',
        'low_surprise':    'light and inquisitive'
    }

    tempo_ranges = {
        'high_negative': (40, 60 + (20 * (1 - stress_score))),
        'mid_negative':  (50, 70 + (10 * (1 - stress_score))),
        'low_negative':  (60, 80 + (10 * stress_score)),
        'high_positive': (100, 140 + (20 * stress_score)),
        'mid_positive':  (90, 120),
        'low_positive':  (70, 90),
        'high_fear':     (50, 70),
        'mid_fear':      (60, 80),
        'low_fear':      (60, 80),
        'high_love':     (70, 90),
        'mid_love':      (60, 80),
        'low_love':      (50, 70),
        'high_surprise': (80, 120),
        'mid_surprise':  (90, 130),
        'low_surprise':  (70, 100)
    }

    tempo_min, tempo_max = tempo_ranges[emotion_type]
    tempo = int(tempo_min + (tempo_max - tempo_min) * stress_score)
    character = therapy_static[emotion_type]

    return (
        f"A perfect loop of {character} instrumental, {tempo} BPM, centered around {freq} Hz drone, "
        "major mode with soft pads, minimal percussion, long reverb, "
        "composed to reduce stress and instill positivity"
    )
prompt=generate_music_prompt(stress_score, emotion)
print(prompt)