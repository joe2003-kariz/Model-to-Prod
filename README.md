# Refund Classifier System

This project classifies refund product images into categories like Shirt, Dress, Jeans, etc. It uses a CNN model served through a Flask API and a batch prediction system that processes images every night.

## 📁 Project Structure

```
refund-classifier/
├── app.py                  # Flask API endpoint
├── batch_predict.py        # Batch prediction script
├── model.h5                # Trained Keras model
├── class_labels.json       # List of categories
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── test.py                 # API test script
├── Batch_Images/           # Input images for batch
├── Batch_Results/          # Output CSV files
└── Refund_Architecture.drawio  # Visual system architecture
```
## How to Run the Project

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Start the Flask API

```bash
python app.py
```

### 3. Test the API

You can use:

* Postman (form-data key: `file`)
* or `test.py`:

```bash
python test_requests.py
```

### 4. Run Batch Prediction

```bash
python batch_predict.py
```

Images in `Batch_Images/` will be processed, and results saved to `Batch_Results/batch_results.csv`.

---

## System Architecture



> The system includes:
>
> * A Flask server for image prediction
> * A nightly batch job
> * A trained TensorFlow/Keras model
> * CSV output for operations review

## Tech Stack

* Python
* Flask
* TensorFlow / Keras
* Git + GitHub
* Windows Task Scheduler

---

## Author

**Your Name**
Law Student & Data Scientist
GitHub: [@joekariuki](https://github.com/joe2003-kariz)

---

## License

This project is for educational purposes.
