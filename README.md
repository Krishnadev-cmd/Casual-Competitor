# 🧠## 🌐 Live Demo

**Try the app now:** [https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/](https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/)

*No installation required! Just upload your CSV file and start discovering causal relationships.*

## 🎥 Demo Video

Watch the app in action! See how to upload data, configure analysis, and discover causal relationships:

https://github.com/user-attachments/assets/9e8b6f48-b2c7-4ddc-8dfc-50078b58ff9e

<details>
<summary>🔧 How to add your Google Drive video</summary>

1. Upload your demo video to Google Drive
2. Right-click the video → "Get shareable link"
3. Make sure it's set to "Anyone with the link can view"
4. Copy the file ID from the URL (the long string between `/d/` and `/view`)
5. Replace `YOUR_GOOGLE_DRIVE_FILE_ID` in the link above with your actual file ID

**Example**: If your Google Drive link is:
`https://drive.google.com/file/d/1ABC123xyz789DEF456/view?usp=sharing`

Then your file ID is: `1ABC123xyz789DEF456`

**Alternative display option** (if you prefer the video to play directly in GitHub):
```html
<div align="center">
  <iframe src="https://drive.google.com/file/d/1xcQMgNFicwbHpBuE5N1mBUsf5P5VdoEJ/view?usp=sharing" 
          width="640" height="360" 
          allow="autoplay">
  </iframe>
</div>
```

</details>

### 🎯 What the Demo Shows:
- **Data Upload**: How to upload and preview your CSV dataset
- **Variable Selection**: Choosing treatment and outcome variables
- **Causal Graph**: Viewing the generated causal relationship graph
- **Inference Results**: Understanding the causal effect estimates
- **Error Handling**: What to do when issues arise
- **Real Example**: Complete walkthrough with sample data

## 🌟 Features Discovery & Inference

A Streamlit web application for discovering causal relationships in your data using Microsoft's DoWhy library. This tool helps you identify and quantify causal effects between variables in your datasets through correlation-based graph discovery and rigorous causal inference.

## � Live Demo

**Try the app now:** [https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/](https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/)

*No installation required! Just upload your CSV file and start discovering causal relationships.*

## �🌟 Features

- **Causal Graph Discovery**: Automatically generates causal graphs using correlation-based analysis
- **Causal Inference**: Estimates causal effects using DoWhy's backdoor identification methods
- **Interactive Web Interface**: User-friendly Streamlit interface with real-time analysis
- **Data Preprocessing**: Automatic data cleaning, encoding, and preprocessing pipeline
- **Fast Mode**: Optimized performance for large datasets
- **Traditional Analysis**: Optional statistical analysis with correlation matrices and visualizations
- **Error Handling**: Comprehensive error handling with helpful debugging information

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)

## 🚀 Quick Start

### 🌐 Option 1: Use the Live Demo (Recommended)
Simply visit: **[https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/](https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/)**

Upload your CSV file and start analyzing causal relationships immediately!

### 💻 Option 2: Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Krishnadev-cmd/Casual-Competitor.git
cd Casual-Competitor
```

2. **Create and activate virtual environment**:
```bash
# Windows
python -m venv venv_py
venv_py\Scripts\activate

# Linux/Mac
python -m venv venv_py
source venv_py/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run locally**:
```bash
streamlit run src/app.py
```

## 📦 Dependencies

- **streamlit**: Web application framework
- **dowhy**: Microsoft's causal inference library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization
- **networkx**: Graph analysis
- **scikit-learn**: Machine learning utilities
- **pydot**: Graph visualization (requires Graphviz)

## 🎯 Usage

### 🌐 Using the Live Demo
1. **Visit**: [https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/](https://casual-competitor-susbk7aqfls5uxswyrp2qz.streamlit.app/)
2. **Upload** your CSV file using the sidebar
3. **Configure** your analysis (see steps below)
4. **Run** and explore the results!

### 💻 Using Local Installation
1. **Start the application**:
```bash
streamlit run src/app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

### 📊 Analysis Configuration (Both Options)
- **Upload** a CSV file
- **Specify treatment variable** (what you want to change)
- **Specify outcome variable** (what you want to predict)
- **Choose** whether to show dataset analysis
- **Enable fast mode** for large datasets
- **Run the analysis** and explore the results!

## 📊 How It Works

### 1. Data Preprocessing
- Automatic ID column removal
- Missing value imputation
- Categorical variable encoding
- Feature scaling and normalization
- Date parsing and feature extraction

### 2. Causal Graph Discovery
- Correlation-based edge detection
- Automatic DAG (Directed Acyclic Graph) construction
- Treatment → Outcome path guarantee
- Threshold-based edge filtering (correlation > 0.3)

### 3. Causal Inference
- DoWhy's backdoor identification method
- Linear regression estimation
- Placebo treatment refutation tests
- Statistical significance testing

## 📁 Project Structure

```
Casual_Competitor/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── doWhy_utils.py      # DoWhy integration utilities
│   ├── Preprocess.py       # Data preprocessing pipeline
│   ├── Utils.py            # Helper functions
│   └── __pycache__/        # Python cache files
├── data/                   # Sample datasets
│   ├── Auto Sales data.csv
│   ├── bakery_Sales.csv
│   ├── big_mart_sales.csv
│   ├── Electronic_sales_Sep2023-Sep2024.csv
│   ├── retail_data.csv
│   └── Video_Games_Sales_as_at_22_Dec_2016.csv
├── venv_py/               # Virtual environment
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration Options

### Sidebar Controls

- **Dataset Analysis**: Show/hide traditional statistical analysis
- **Preprocessing**: Specify if your dataset is already preprocessed
- **Fast Mode**: Enable for faster processing on large datasets
- **Treatment Variable**: The variable you want to manipulate
- **Outcome Variable**: The variable you want to predict/analyze

### Performance Tips

- **Fast Mode**: Recommended for datasets with >500 rows or >15 columns
- **Preprocessing**: Skip if your data is already clean and encoded
- **Variable Selection**: Choose variables with clear causal relationships

## 📈 Example Use Cases

1. **Marketing Analysis**:
   - Treatment: Marketing spend
   - Outcome: Sales revenue
   - Discover: How marketing investment affects sales

2. **Pricing Strategy**:
   - Treatment: Product price
   - Outcome: Customer demand
   - Discover: Price elasticity effects

3. **A/B Testing**:
   - Treatment: Feature flag (0/1)
   - Outcome: User engagement
   - Discover: Feature impact on user behavior

## ⚠️ Important Notes

- **Causal vs Correlation**: This tool identifies potential causal relationships, but domain expertise is still required for interpretation
- **Data Quality**: Results are only as good as your input data - ensure clean, relevant datasets
- **Sample Size**: Larger datasets generally produce more reliable causal estimates
- **Variable Selection**: Choose treatment and outcome variables with theoretical causal relationships

## 🐛 Troubleshooting

### Common Issues

1. **"Variable not found"**: Check that your treatment/outcome variables exactly match column names
2. **"Graph contains cycles"**: Enable fast mode or check for circular relationships in your data
3. **Zero causal effect**: May indicate no causal relationship or insufficient data signal
4. **Visualization errors**: Ensure Graphviz is installed for graph plotting

### Debug Information

The app provides comprehensive debug information including:
- Graph structure visualization
- Available column names
- Raw inference results
- Error details and suggestions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft DoWhy**: For the powerful causal inference framework
- **Streamlit**: For the excellent web application framework
- **NetworkX**: For graph analysis capabilities
- **Pandas & NumPy**: For data manipulation foundations

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Look at the debug information in the app
3. Open an issue on GitHub
4. Review the DoWhy documentation for advanced usage

## 🚀 Future Enhancements

- [ ] Support for more causal discovery algorithms
- [ ] Advanced visualization options
- [ ] Export functionality for results
- [ ] Batch processing capabilities
- [ ] Integration with more causal inference methods
- [ ] Real-time data streaming support

---

**Happy Causal Discovery! 🎉**
