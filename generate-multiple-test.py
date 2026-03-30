"""
Generate two test PDFs on different ML/Stats topics.
Run: python generate_test_pdfs.py
"""

import fitz
import os

os.makedirs("data/raw_pdfs", exist_ok=True)


def create_pdf(filename: str, pages: list[dict]):
    doc = fitz.open()
    for page_data in pages:
        page = doc.new_page(width=612, height=792)
        title_rect = fitz.Rect(50, 40, 562, 80)
        page.insert_textbox(title_rect, page_data["title"], fontsize=16, fontname="helv", color=(0.1, 0.1, 0.5))
        page.draw_line(fitz.Point(50, 85), fitz.Point(562, 85), color=(0.3, 0.3, 0.3), width=0.5)
        content_rect = fitz.Rect(50, 95, 562, 750)
        page.insert_textbox(content_rect, page_data["content"], fontsize=10, fontname="helv", color=(0, 0, 0))
    path = f"data/raw_pdfs/{filename}"
    doc.save(path)
    doc.close()
    print(f"Created: {path} ({len(pages)} pages)")


# ─────────────────────────────────────────────
# PDF 1: Probability and Statistics Foundations
# ─────────────────────────────────────────────
stats_pages = [
    {
        "title": "Chapter 1: Probability Basics",
        "content": """Probability is the mathematical framework for quantifying uncertainty. It assigns a number 
between 0 and 1 to events, where 0 means impossible and 1 means certain.

The sample space (S) is the set of all possible outcomes of an experiment. An event is a subset of the 
sample space. For example, when rolling a die, the sample space is {1, 2, 3, 4, 5, 6} and the event 
"rolling an even number" is {2, 4, 6}.

Axioms of Probability (Kolmogorov):
1. P(A) >= 0 for any event A
2. P(S) = 1 for the sample space S
3. For mutually exclusive events A and B: P(A or B) = P(A) + P(B)

Conditional Probability measures the probability of an event given that another event has occurred:
P(A|B) = P(A and B) / P(B), provided P(B) > 0.

Bayes' Theorem relates conditional probabilities and is fundamental to machine learning:
P(A|B) = P(B|A) * P(A) / P(B)

This allows us to update our beliefs (prior probability P(A)) after observing evidence (B) to get a 
posterior probability P(A|B). The term P(B|A) is called the likelihood, and P(B) is the evidence or 
marginal likelihood.

Independence: Two events A and B are independent if P(A and B) = P(A) * P(B). Equivalently, 
knowing B occurred does not change the probability of A: P(A|B) = P(A)."""
    },
    {
        "title": "Chapter 2: Random Variables and Distributions",
        "content": """A random variable is a function that assigns a numerical value to each outcome in a sample 
space. Random variables can be discrete (countable values) or continuous (any value in an interval).

The Expected Value (mean) of a discrete random variable X is:
E[X] = sum of x * P(X = x) for all possible values x.
It represents the long-run average value.

Variance measures the spread of a distribution around its mean:
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
The standard deviation is the square root of the variance.

Important Discrete Distributions:

Bernoulli Distribution: Models a single trial with two outcomes (success/failure). P(X=1) = p, P(X=0) = 1-p.
Mean = p, Variance = p(1-p).

Binomial Distribution: Models the number of successes in n independent Bernoulli trials.
P(X=k) = C(n,k) * p^k * (1-p)^(n-k). Mean = np, Variance = np(1-p).

Poisson Distribution: Models the number of events occurring in a fixed interval of time or space.
P(X=k) = (lambda^k * e^(-lambda)) / k!. Mean = lambda, Variance = lambda.

Important Continuous Distributions:

Normal (Gaussian) Distribution: The most important continuous distribution. Defined by mean (mu) and 
variance (sigma^2). The standard normal has mu=0 and sigma=1. Approximately 68% of values fall within 
one standard deviation of the mean, 95% within two, and 99.7% within three (the 68-95-99.7 rule).

Exponential Distribution: Models the time between events in a Poisson process. 
Mean = 1/lambda, Variance = 1/lambda^2. It has the memoryless property: P(X > s+t | X > s) = P(X > t)."""
    },
    {
        "title": "Chapter 3: Statistical Inference",
        "content": """Statistical inference is the process of drawing conclusions about a population from a sample.

Point Estimation: A point estimate is a single value used to estimate a population parameter. 
The sample mean (x-bar) estimates the population mean (mu). The sample variance (s^2) estimates 
the population variance (sigma^2).

Properties of good estimators:
- Unbiasedness: E[estimator] = true parameter value
- Consistency: As sample size increases, the estimator converges to the true value
- Efficiency: Among unbiased estimators, the one with lowest variance is most efficient

Confidence Intervals provide a range of plausible values for a parameter. A 95% confidence interval 
means that if we repeated the experiment many times, 95% of the intervals would contain the true 
parameter. For a population mean with known variance: x-bar +/- z*(sigma/sqrt(n)), where z* is 
the critical value (1.96 for 95%).

Hypothesis Testing: A formal procedure for deciding between two competing claims.
- Null hypothesis (H0): The default assumption (e.g., no effect, no difference)
- Alternative hypothesis (H1): What we want to show (e.g., there is an effect)
- Test statistic: A value computed from the sample data
- p-value: Probability of observing data as extreme as ours, assuming H0 is true
- If p-value < significance level (alpha, typically 0.05), we reject H0

Type I Error (False Positive): Rejecting H0 when it is actually true. Probability = alpha.
Type II Error (False Negative): Failing to reject H0 when H1 is true. Probability = beta.
Power = 1 - beta = probability of correctly rejecting a false H0.

The Central Limit Theorem (CLT) states that the distribution of the sample mean approaches a normal 
distribution as the sample size increases, regardless of the population distribution. This is why 
the normal distribution is so important in statistics and why many statistical tests work even when 
the underlying data is not normally distributed."""
    },
    {
        "title": "Chapter 4: Maximum Likelihood Estimation",
        "content": """Maximum Likelihood Estimation (MLE) is the most widely used method for estimating parameters 
of a statistical model. The idea is to find the parameter values that make the observed data most probable.

Given observed data x1, x2, ..., xn and a model with parameter theta, the likelihood function is:
L(theta) = P(x1, x2, ..., xn | theta) = product of P(xi | theta) for independent observations.

In practice, we maximize the log-likelihood (easier to work with):
log L(theta) = sum of log P(xi | theta)

To find the MLE, take the derivative of the log-likelihood with respect to theta, set it equal to zero, 
and solve. For example, the MLE of the mean of a normal distribution is the sample mean x-bar.

Properties of MLE:
- Consistent: Converges to the true parameter as n increases
- Asymptotically normal: For large n, the MLE is approximately normally distributed
- Asymptotically efficient: Achieves the lowest possible variance for large samples
- Invariant: If theta-hat is the MLE of theta, then g(theta-hat) is the MLE of g(theta)

MLE is the foundation for many machine learning methods. Logistic regression, for example, finds 
parameters by maximizing the likelihood of the observed class labels. Similarly, fitting a Gaussian 
Mixture Model uses MLE through the Expectation-Maximization (EM) algorithm.

The EM Algorithm is used when the data has latent (hidden) variables:
E-step: Compute the expected value of the log-likelihood given current parameter estimates
M-step: Maximize this expected log-likelihood to update parameters
Repeat until convergence.

The EM algorithm is guaranteed to increase the likelihood at each step, though it may converge to 
a local maximum rather than the global maximum."""
    },
]

# ─────────────────────────────────────────────
# PDF 2: Natural Language Processing
# ─────────────────────────────────────────────
nlp_pages = [
    {
        "title": "Chapter 1: Text Preprocessing",
        "content": """Natural Language Processing (NLP) is the field of AI concerned with enabling computers to 
understand, interpret, and generate human language. Text preprocessing is the critical first step in 
any NLP pipeline.

Tokenization is the process of splitting text into individual units (tokens). These can be words, 
subwords, or characters depending on the approach.
- Word tokenization: "The cat sat" -> ["The", "cat", "sat"]
- Subword tokenization (BPE): Breaks rare words into common subword units. Used by GPT and BERT.
- Character tokenization: Splits into individual characters. Handles any vocabulary but loses word meaning.

Lowercasing converts all text to lowercase to reduce vocabulary size, though it can lose information 
(e.g., "Apple" the company vs "apple" the fruit).

Stop Word Removal eliminates common words like "the", "is", "at" that carry little meaning. 
This is useful for bag-of-words models but can hurt models that rely on word order.

Stemming reduces words to their root form by removing suffixes: "running" -> "run", "better" -> "bet". 
The Porter Stemmer is the most common algorithm. It is fast but can produce non-real words.

Lemmatization is more sophisticated than stemming. It reduces words to their dictionary form (lemma) 
using morphological analysis: "better" -> "good", "ran" -> "run". SpaCy and NLTK provide lemmatization.

Regular Expressions (regex) are patterns used to match and manipulate text. They are essential for 
text cleaning tasks like removing HTML tags, extracting email addresses, or normalizing whitespace."""
    },
    {
        "title": "Chapter 2: Text Representation",
        "content": """Converting text to numerical representations is essential for machine learning models.

Bag of Words (BoW): Represents a document as a vector of word counts. Each dimension corresponds to a 
unique word in the vocabulary. BoW ignores word order — "dog bites man" and "man bites dog" have the 
same representation. The vocabulary can be very large, leading to sparse, high-dimensional vectors.

TF-IDF (Term Frequency - Inverse Document Frequency): Improves on BoW by weighting words based on 
their importance. TF measures how often a word appears in a document. IDF measures how rare a word 
is across all documents. TF-IDF = TF * IDF. Words that are frequent in one document but rare overall 
get high scores, while common words like "the" get low scores.

Word Embeddings map words to dense, low-dimensional vectors where similar words are close together.

Word2Vec (Mikolov et al., 2013) learns embeddings using neural networks in two architectures:
- CBOW (Continuous Bag of Words): Predicts a target word from surrounding context words.
- Skip-gram: Predicts context words from a target word. Better for rare words.
Word2Vec captures semantic relationships: vector("king") - vector("man") + vector("woman") ≈ vector("queen").

GloVe (Global Vectors): Learns embeddings from word co-occurrence statistics across the entire corpus. 
Combines the benefits of count-based methods and prediction-based methods like Word2Vec.

FastText: Extends Word2Vec by representing words as bags of character n-grams. This allows it to 
generate embeddings for out-of-vocabulary words by combining the n-gram vectors. For example, 
"unhappiness" can be represented even if never seen during training by combining embeddings for 
"un", "hap", "happy", "ness", etc."""
    },
    {
        "title": "Chapter 3: Sequence Models",
        "content": """Many NLP tasks require understanding the sequential nature of language.

Recurrent Neural Networks (RNNs) process sequences one element at a time, maintaining a hidden state 
that captures information from previous time steps. At each step t, the hidden state h_t is computed as:
h_t = activation(W_h * h_(t-1) + W_x * x_t + b)

The Vanishing Gradient Problem: During backpropagation through time, gradients can become exponentially 
small, making it difficult for standard RNNs to learn long-range dependencies. If a relevant word 
appeared 50 steps ago, the gradient signal may vanish before reaching it.

Long Short-Term Memory (LSTM) networks solve this with a gating mechanism:
- Forget gate: Decides what information to discard from the cell state
- Input gate: Decides what new information to store in the cell state
- Output gate: Decides what to output based on the cell state
The cell state acts as a highway for information, allowing gradients to flow unchanged over many steps.

Gated Recurrent Units (GRUs) are a simplified version of LSTMs with only two gates:
- Reset gate: Controls how much past information to forget
- Update gate: Controls how much new information to add
GRUs have fewer parameters than LSTMs and often perform comparably.

Bidirectional RNNs process the sequence in both forward and backward directions, capturing both 
past and future context. The outputs from both directions are concatenated at each time step. 
This is particularly useful for tasks like named entity recognition where both left and right 
context matter.

Sequence-to-Sequence (Seq2Seq) models use an encoder-decoder architecture:
- Encoder: Reads the input sequence and compresses it into a fixed-length context vector
- Decoder: Generates the output sequence from the context vector
Applications include machine translation, text summarization, and chatbots."""
    },
    {
        "title": "Chapter 4: Attention and Transformers",
        "content": """The Attention Mechanism was introduced to address the bottleneck of compressing an entire input 
sequence into a single fixed-length vector in Seq2Seq models.

Instead of relying solely on the final encoder hidden state, attention allows the decoder to look back 
at all encoder hidden states and focus on the most relevant ones for each output token. The attention 
score between a decoder state and each encoder state determines how much focus to place on each input 
position.

Types of Attention:
- Additive Attention (Bahdanau): Uses a learned feedforward network to compute alignment scores.
- Dot-Product Attention: Computes scores as the dot product of query and key vectors. Faster but can 
  have scaling issues with high dimensions.
- Scaled Dot-Product Attention: Divides dot-product scores by sqrt(d_k) to stabilize gradients.

The Transformer architecture (Vaswani et al., 2017) replaced RNNs entirely with self-attention:

Self-Attention allows each token in a sequence to attend to every other token, capturing relationships 
regardless of distance. Each token is projected into three vectors: Query (Q), Key (K), and Value (V). 
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Multi-Head Attention runs multiple attention operations in parallel, allowing the model to attend to 
information at different positions from different representation subspaces.

Positional Encoding: Since transformers have no recurrence, they need explicit position information. 
Sinusoidal functions of different frequencies encode the position of each token.

The Transformer Block consists of: Multi-Head Self-Attention -> Add & Normalize -> Feed-Forward Network 
-> Add & Normalize. Multiple blocks are stacked to create deep models.

BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder part of the 
transformer. It is pre-trained on masked language modeling (predicting randomly hidden words) and 
next sentence prediction. BERT revolutionized NLP by providing pre-trained representations that can 
be fine-tuned for downstream tasks like sentiment analysis, question answering, and named entity recognition.

GPT (Generative Pre-trained Transformer) uses only the decoder part with causal (left-to-right) 
attention. It is pre-trained on next token prediction and excels at text generation tasks."""
    },
]

create_pdf("stats_foundations.pdf", stats_pages)
create_pdf("nlp_fundamentals.pdf", nlp_pages)

print("\n" + "=" * 60)
print("TEST QUESTIONS")
print("=" * 60)

print("\n--- Stats PDF (in-scope) ---")
print('python main.py ask "What is Bayes theorem?"')
print('python main.py ask "Explain the Central Limit Theorem"')
print('python main.py ask "What is a p-value?"')
print('python main.py ask "How does maximum likelihood estimation work?"')
print('python main.py ask "What is the EM algorithm?"')

print("\n--- NLP PDF (in-scope) ---")
print('python main.py ask "What is tokenization?"')
print('python main.py ask "How does TF-IDF work?"')
print('python main.py ask "What is the vanishing gradient problem?"')
print('python main.py ask "Explain the attention mechanism"')
print('python main.py ask "What is the difference between BERT and GPT?"')

print("\n--- Cross-document ---")
print('python main.py ask "How is Bayes theorem used in NLP?"')

print("\n--- Out-of-scope ---")
print('python main.py ask "How do I make pasta?"')
print('python main.py ask "What is blockchain?"')