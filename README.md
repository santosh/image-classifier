# image-classifer

This is a simple example of a image classification by using our own images for training. The training data used is for a parking lot, where the images are divided into two classes: empty and occupied.

## Installation

To install the necessary packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

There is a single monolithic file which does the following:

1. Load the images from the `data` directory.
2. Preprocess the images.
3. Split the images into training and testing sets.
4. Train the model.
5. Evaluate the model.
6. Save the model.

To run it, run the following command:

```bash
python main.py
```

You'll see an output like this:

```
99.91% of samples were correctly classified.
```

## Contributing

TODO: Use images outside the `data` directory to inference.

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
