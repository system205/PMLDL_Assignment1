# Path in solution creation process

## Data

For this dataset it was extremely important to preprocess the data correctly. First, I explored the values in dataset and noticed that some translations have higher toxicity score then the according reference. So, I flipped them.

After it I added useful statistical columns that will help me to split the dataset before training and visualize it in a more readable way. The whole preprocessing pipeline is the following:

1. Load dataframe
1. Swap reference with translation if toxicity is wrong
1. Add columns:
    - toxicity difference
    - reference length
    - translation length
    - difference in lengths. The initial lenght_diff is not the same
1. Round columns to sort by them easier:
    - similarity, 2 decimal points
    - lenght_diff, 2 decimal points
1. Preprocess translation and reference sentences:
    - lower_text
    - remove_numbers
    - remove_punctuation
    - remove_multiple_spaces
1. Remove sentences with unknown symbols to ease the training

## Ideas

By exploring and analyzing the dataset I understood that there are a lot of swearing words that make the sentence toxic. So, it was the goal number one to get rid of them.

### Hypothesis 1

So, the first idea appeared: extract words that are only in references, combine with the well-known english bad words and compose a model that cleans the sentence from these words. It became the Hypothesis 1.

### Hypothesis 2

Another idea was to use nlp or large language model. Previously I have heard of the use of T5 model. So, I wanted to fine tune it on text paraphrasing. This approach seems more robust and workable since LLM pay attention to the context as well.

### Metric

I used **'sacrebleu'** as the metric for comparison the paraphrased output text with the target translation. This metric is used to asses the similarity between machine translated text and the target one. The paraphrasing in our task is similar to translation: from toxic language to normal english. It was another **idea**, to prompt the model like 'translate from toxic to normal'.

## Problems

There is several problems that appeared during the whole research.

1. The translation and toxic score assignment are very subjective. I didn't understand the toxicity behind some very toxic references. There were no swearing in them at least. In addition, some translations seemed not correct in my opinion. This also leads to the next problem.

1. The meaning might not preserve. Paraphrasing is a complicated task since we change the syntax but have to save the semantics. 
    - First, there might be several "true" paraphrases. The presence only a single translation make the model to strictly follow it
    - Second, the meaning is different for different people as it may concern religion, location, dialects etc. So, it is nearly impossible to paraphrase preserving the meaning equally to everyone.
1. The dataset is large. The number of sentences is way more than 100K. In addition, not all references are severely toxic. So, to ease the training process and focus on the removing toxic words or paraphrasing them I picked a subset with the following criteria:
    - The sentences themselves are not long. Less than 70 characters
    - Pick very toxic input. Reference toxicity is above 0.95
    - Get very neutral output. Translation toxicity is below 0.002
    - The meaning should be preserved enough. The similarity is above 0.8
    - Difference in lengths between reference and translation is less than 15 characters. Again for similarity.

1. Since I preprocessed the dataset the output sentences are preprocessed as well. So, they look not so good and the future postprocessing is required to make the output more readable. Example: i don t like london -> I don't like London.

## Baseline: Straight translation

First of all, we need to know the zero level metric below which it is prohibited to go. The simplest way to paraphrase is to replace all words in a toxic sentence by the same words. In this case the meaning will be 100% preserved but the toxicity level stays the same as well.
Even though it is a straightforward translation, evaluating the whole dataset will give a clue on the similarity between translated and toxic sentences.

## Results

### Baseline result

I've tested the baseline model on 3 types of datasets:

- The whole initial dataset: 22%
- Preprocessed dataset: 25%
- Train dataset: **37%**

So, when constructing the next solutions I compared them with the highest metric of Baseline.

### Hypothesis 1 result

For analyzing the presence of bad words I used only train dataset as it has the most toxic references and the least toxic translations.

The resulting sacreBLEU score on validation dataset is the following: 45%. **Which is 8% more** than baseline result.

### Hypothesis 2 result

From the first glance fine tuned T5 performed well. At least, it removes swearing.

The validation sacreBLEU score is 53%



