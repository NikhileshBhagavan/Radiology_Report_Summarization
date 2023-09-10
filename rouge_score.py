from rouge import Rouge
rouge = Rouge()


def calculate_rouge_scores(generated_summaries, reference_summaries):

    scores = rouge.get_scores(
        generated_summaries, reference_summaries, avg=True)
    print("ROUGE1_SCORE : ", scores['rouge-1']['f'])
    print("ROUGE2_SCORE : ", scores['rouge-2']['f'])
    print("ROUGEL_SCORE : ", scores['rouge-l']['f'])
    return {'rouge-1': scores['rouge-1']['f']*100, 'rouge-2': scores['rouge-2']['f']*100, 'rouge-l': scores['rouge-l']['f']*100}
