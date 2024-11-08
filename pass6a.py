import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(position, d_model):
    """ Generate positional encoding for a given position and model dimensionality. """
    encoding = np.zeros((d_model,))
    for k in range(d_model):
        if k % 2 == 0:
            encoding[k] = np.sin(position / (10000 ** (2 * k / d_model)))
        else:
            encoding[k] = np.cos(position / (10000 ** (2 * k / d_model)))
    return encoding

# Sentence with 10 words
sentence = "The quick brown fox jumps over the lazy dog back"
words = sentence.split()

# Dimension model
d_model = 4
d_model = 20

# Calculate positional encodings for each word
positional_encodings = []
for i, word in enumerate(words):
    encoding = positional_encoding(i+1, d_model)
    positional_encodings.append(encoding)
    print(f"Position {i+1} ({word}): {encoding}")


# Plotting
plt.figure(figsize=(12, 8))
for i, encoding in enumerate(positional_encodings):
    plt.plot(range(1, d_model+1), encoding, marker='o', label=f"{words[i]}")

plt.title('Positional Encoding Values Across Dimensions for Each Word')
plt.xlabel('Dimension')
plt.ylabel('Encoding Value')
plt.xticks(range(1, d_model+1))
plt.grid(True)
plt.legend(title='Words', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

