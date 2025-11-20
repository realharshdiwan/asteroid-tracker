# Contributing to NASA Asteroid Tracker

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git
- NASA API key (optional, for testing)

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/asteroid-tracker.git
cd asteroid-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your NASA API key

# Run the app
streamlit run app.py
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_app.py::TestRiskCalculation -v
```

---

## 📝 Code Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small
- Comment complex logic

### Example

```python
def calculate_risk_score(diameter, distance, velocity, weights):
    """
    Calculate risk score for an asteroid.
    
    Args:
        diameter: Asteroid diameter in meters
        distance: Miss distance in kilometers
        velocity: Relative velocity in km/h
        weights: Tuple of (w_diameter, w_distance, w_velocity)
    
    Returns:
        Risk score between 0 and 1
    """
    # Implementation
    pass
```

---

## 🔀 Contribution Workflow

1. **Create an issue** describing the bug or feature
2. **Fork the repository**
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Add tests** for new functionality
6. **Run tests** to ensure nothing breaks
7. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Feature description"
   ```
8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Create a Pull Request**

---

## 🐛 Reporting Bugs

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details (OS, Python version)

---

## 💡 Suggesting Features

Include:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (optional)
- Mockups or examples (if UI-related)

---

## 📚 Areas for Contribution

### High Priority
- [ ] Additional visualizations (3D orbit plots, impact probability)
- [ ] Email notification support
- [ ] Export to additional formats (PDF, Excel)
- [ ] Mobile-responsive improvements
- [ ] Performance optimizations

### Medium Priority
- [ ] Historical data analysis
- [ ] Comparison with previous approaches
- [ ] Multi-language support
- [ ] Dark/light theme toggle
- [ ] Advanced filtering options

### Documentation
- [ ] Video tutorials
- [ ] API client examples (more languages)
- [ ] Architecture diagrams
- [ ] Performance benchmarks

---

## ✅ Pull Request Checklist

Before submitting:
- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Screenshots included (if UI changes)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!
