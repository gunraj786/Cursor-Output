# 🚀 Gemini AI Assistant

A comprehensive Streamlit application that harnesses Google's Gemini AI with advanced features including parallel processing, vision analysis, analytics dashboard, and LangChain integration.

## ✨ Features

### 🧠 Chat Assistant
- Interactive AI conversations with Gemini
- Pre-built prompt templates for various use cases
- Real-time response metrics and analytics
- Downloadable responses

### ⚡ Batch Processing  
- Process multiple prompts simultaneously
- Concurrent execution for improved efficiency
- File upload support for bulk prompts
- Progress tracking and success metrics

### 🖼️ Vision Analysis
- Upload and analyze images with Gemini Vision
- Multiple analysis types (description, object detection, OCR)
- Detailed image metadata display
- Export analysis results

### 📊 Analytics Dashboard
- Comprehensive usage analytics
- Interactive charts and visualizations
- Performance metrics tracking
- Data export capabilities (CSV/JSON)

### 🔗 LangChain Integration
- Advanced prompt engineering with templates
- Business analysis, content creation, and educational templates
- Custom template support with variable substitution
- Chain execution with output parsing

### 📊 Optional Langfuse Tracking
- Advanced logging and tracing
- Performance monitoring
- Usage analytics and insights

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google AI API key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd gemini-ai-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run streamlit_gemini_assistant.py
   ```

4. **Configure API Key:**
   - Open the application in your browser
   - Enter your Google AI API key in the sidebar
   - Start using the AI tools!

## 🚀 Quick Start

1. **Get your API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Launch the app** with `streamlit run streamlit_gemini_assistant.py`
3. **Enter your API key** in the sidebar configuration
4. **Choose a tool** from the navigation menu
5. **Start creating** with AI!

## 🎯 Usage Examples

### Chat Assistant
```
Template: Business Analysis
Prompt: "Analyze the current state of the electric vehicle market and provide strategic recommendations for a new startup"
```

### Batch Processing
```
Input multiple prompts:
- Explain quantum computing
- Write a haiku about technology  
- List benefits of renewable energy
- Create a marketing slogan for eco-friendly products
```

### Vision Analysis
```
Upload an image and ask:
- "Describe this image in detail"
- "Identify all objects in this photo"
- "Extract any text visible in this image"
```

### LangChain Templates
```
Business Analysis Template:
Company: TechStart Inc
Industry: SaaS
Challenge: Customer retention

→ Generates comprehensive business analysis
```

## 📊 Analytics Features

- **Response Time Tracking**: Monitor AI response performance
- **Token Usage**: Track input/output token consumption  
- **Success Metrics**: Batch processing success rates
- **Usage Trends**: Visualize usage patterns over time
- **Export Capabilities**: Download data as CSV or JSON

## 🔧 Configuration Options

### Sidebar Settings
- **Google API Key**: Required for all AI functionality
- **Langfuse Integration**: Optional advanced tracking
- **Session Statistics**: Real-time usage metrics

### Advanced Features
- **Concurrent Processing**: Configurable worker threads
- **Temperature Control**: Adjust AI creativity levels
- **Template Variables**: Dynamic prompt customization
- **Export Options**: Multiple download formats

## 🛡️ Error Handling

The application includes comprehensive error handling for:
- Invalid API keys
- Network connectivity issues
- File upload problems
- Processing timeouts
- Invalid input formats

## 📈 Performance Optimization

- **Parallel Processing**: Concurrent API calls for batch operations
- **Caching**: Streamlit caching for improved performance  
- **Session State**: Efficient data management across sessions
- **Progress Tracking**: Real-time feedback for long operations

## 🔒 Privacy & Security

- API keys are handled securely through environment variables
- No data is stored permanently on the server
- Session data is cleared when the browser is closed
- Optional Langfuse integration for advanced tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

For issues, questions, or feature requests:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

## 🔄 Updates

The application is regularly updated with:
- New AI model integrations
- Enhanced UI/UX features
- Performance improvements
- Bug fixes and security updates

---

**Built with ❤️ using Streamlit and Google's Gemini AI**
