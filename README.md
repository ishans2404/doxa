# Doxa

A modular, intelligent machine learning library with automatic device management and unified algorithm interfaces.

## 🌟 Features

- **Intelligent Device Management**: Automatic CPU/GPU selection based on data size and hardware
- **Unified Algorithm Interface**: Consistent API across all ML categories
- **Plugin Architecture**: Easy extensibility without modifying core code
- **Memory-Efficient Processing**: Lazy evaluation and streaming for large datasets
- **Cross-Platform Support**: Windows, Linux, macOS compatibility

## 🚀 Quick Start

```python
import doxa as dx

# Automatic device selection
x = dx.tensor([[1, 2, 3], [4, 5, 6]])
y = dx.tensor([1, 0])

# Unified interface for all algorithms
model = dx.algorithms.regression.LinearRegression()
model.fit(x, y)
predictions = model.predict(x)
```

## 📦 Installation

```bash
pip install doxa
```

For GPU support:
```bash
pip install doxa[gpu]
```

## 🛠️ Development

This project is in active development. Current status: Foundation phase.

## 📖 Documentation

Coming soon!

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines.

## 📄 License

[MIT License](https://github.com/ishans2404/doxa/blob/main/LICENSE)
