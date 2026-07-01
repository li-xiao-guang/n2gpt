import json

import numpy as np
import requests

from src.dataset import Dataset
from src.gpt import GPT
from src.loss import CELoss
from src.optim import AdamOptimizer, WarmupCosineScheduler
from src.tensor import Tensor

DATA_FILE = 'tinyshakespeare.txt'
MODEL_FILE = 'tinyshakespeare.npz'


def download_dataset():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(DATA_FILE, 'w') as f:
        f.write(response.text)


class LLMDataset(Dataset):

    def __init__(self, filename, context_size=64, stride=None, train_split=0.9):
        self.filename = filename
        self.context_size = context_size
        self.stride = stride if stride is not None else context_size // 2
        self.train_split = train_split

        self.vocabulary = []
        self.stoi = {}
        self.itos = {}
        self.tokens = []

        self.train_tokens = []
        self.eval_tokens = []
        super().__init__(batch_size=1)

    def load(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            text = f.read()

        self.vocabulary = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocabulary)}
        self.itos = {i: ch for i, ch in enumerate(self.vocabulary)}
        self.tokens = self.encode(text)

        split = int(len(self.tokens) * self.train_split)
        self.train_tokens = self.tokens[:split]
        self.eval_tokens = self.tokens[split:]

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def _build_samples(self, tokens):
        features = []
        s = self.context_size
        for i in range(0, len(tokens) - s - 1, self.stride):
            x = tokens[i: i + s]
            y = tokens[i + 1: i + s + 1]
            features.append((x, y))
        return features

    def train(self):
        self.train_features = self._build_samples(self.train_tokens)
        self.features = self.train_features
        self.labels = None

    def eval(self):
        self.test_features = self._build_samples(self.eval_tokens)
        self.features = self.test_features
        self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x, y = self.features[index]
        return Tensor(x), Tensor(self.onehot(y))

    def shape(self):
        return (self.context_size,), (self.context_size, self.vocab_size)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return "".join(self.itos[i] for i in tokens)

    def onehot(self, tokens):
        ebd = np.zeros((len(tokens), self.vocab_size), dtype=np.float32)
        ebd[np.arange(len(tokens)), tokens] = 1
        return ebd

    @staticmethod
    def argmax(vector):
        return int(np.argmax(vector))

    def estimate(self, predictions):
        count = 0
        for i in range(len(predictions)):
            prediction = self.argmax(predictions[i].data[-1])
            _, y = self.features[i]
            target = y[-1]
            if prediction == target:
                count += 1
        return count / len(predictions)


class Model:

    def __init__(self, layer, loss, optimizer):
        self.layer = layer
        self.loss = loss
        self.optimizer = optimizer

    def train(self, dataset, epochs, scheduler=None):
        self.layer.train()
        dataset.train()
        steps = 0
        order = list(range(len(dataset)))

        for epoch in range(epochs):
            np.random.shuffle(order)
            loss = 0.0

            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.lr_at(steps)

                feature, label = dataset[i]
                prediction = self.layer(feature)
                error = self.loss(prediction, label)

                self.optimizer.reset()
                error.backward()
                loss += float(error.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                steps += 1

                if (step + 1) % 100 == 0:
                    lr_str = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(loss / 100):.4f}{lr_str}")
                    loss = 0.0

    def test(self, dataset):
        self.layer.eval()
        dataset.eval()

        predictions = []
        for i in range(len(dataset)):
            feature, label = dataset[i]
            prediction = self.layer(feature)
            predictions.append(prediction)
        return predictions


CONTEXT_SIZE = 128
EMBEDDING_SIZE = 256
HEADS = 8
LAYERS = 8
FFN_HIDDEN = 512
DROPOUT = 0.1

MAX_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
EPOCHS = 15


def save_model(path, layer, config):
    params = {f"param_{i}": p.data for i, p in enumerate(layer.parameters())}
    np.savez(path, config=json.dumps(config), **params)
    print(f"saved model to {path} ({len(params)} parameter tensors)")


def load_model(path):
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data['config']))
    layer = GPT(**config)
    for i, p in enumerate(layer.parameters()):
        p.data = data[f"param_{i}"].astype(p.data.dtype)
        p.grad = np.zeros_like(p.data)
    return layer, config


def generate(layer, dataset, prompt, new_tokens=200, temperature=0.7, top_k=20):
    tokens = dataset.encode(prompt)
    context_size = dataset.context_size

    layer.eval()
    for _ in range(new_tokens):
        window = tokens[-context_size:]
        feature = Tensor(window)
        logits = layer.forward(feature)
        last_logits = logits.data[-1, :].astype(np.float64) / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, last_logits.shape[-1])
            threshold = np.partition(last_logits, -k)[-k]
            last_logits[last_logits < threshold] = -np.inf

        exp = np.exp(last_logits - np.max(last_logits))
        probs = exp / np.sum(exp)

        token = np.random.choice(len(probs), p=probs)
        tokens.append(int(token))

    return dataset.decode(tokens)


def main():
    dataset = LLMDataset(DATA_FILE, context_size=CONTEXT_SIZE)
    print(f"vocab_size={dataset.vocab_size} train_samples={len(dataset.train_features)}")

    config = dict(
        vocabulary_size=dataset.vocab_size,
        context_size=CONTEXT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        heads=HEADS,
        layers=LAYERS,
        ffn_hidden=FFN_HIDDEN,
        dropout=DROPOUT
    )
    layer = GPT(**config)
    num_params = sum(p.data.size for p in layer.parameters())
    print(f"parameters={num_params}")

    loss_fn = CELoss()
    optimizer = AdamOptimizer(layer.parameters(), lr=MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(dataset.train_features)
    scheduler = WarmupCosineScheduler(
        max_lr=MAX_LEARNING_RATE,
        total_steps=total_steps,
        warmup_steps=WARMUP_STEPS,
        min_lr=MIN_LEARNING_RATE,
    )

    model = Model(layer, loss_fn, optimizer)
    model.train(dataset, EPOCHS, scheduler=scheduler)

    save_model(MODEL_FILE, layer, config)

    predictions = model.test(dataset)
    print(f"Accuracy: {dataset.estimate(predictions):.3f}")

    prompt = "ROMEO:"
    text = generate(layer, dataset, prompt=prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text}")


if __name__ == '__main__':
    main()
