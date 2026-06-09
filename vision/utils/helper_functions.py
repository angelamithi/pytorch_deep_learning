"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path

import requests

import pathlib
import torch

from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict,Tuple

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path
from pathlib import Path
from typing import Callable, Tuple, Optional
from torch.utils.data import Dataset


def download_torchvision_dataset(
    dataset_class,
    root: str = "data",
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    split_type="train_test",
    download: bool = True,
) -> Tuple[Dataset, Dataset]:


    """
    Downloads and prepares a torchvision dataset with train/test splits.

    Args:
        dataset_class: torchvision dataset class (e.g. datasets.Food101)
        root (str): Root directory to store data
        train_transform (Callable): Transform for training data
        test_transform (Callable): Transform for test data
        split_type (str): Type of split argument used by dataset
            (e.g. "train/test" or "standard")
        download (bool): Whether to download dataset

    Returns:
        Tuple[train_dataset, test_dataset]
    """

    data_dir = Path(root)

    # -------------------------------------------------------
    # Handle Food101-style datasets (train/test split)
    # -------------------------------------------------------
    if split_type == "train_test":

        train_data = dataset_class(
            root=data_dir,
            split="train",
            transform=train_transform,
            download=download
        )

        test_data = dataset_class(
            root=data_dir,
            split="test",
            transform=test_transform,
            download=download
        )

    # -------------------------------------------------------
    # Handle datasets that use train=True/False (e.g. CIFAR10)
    # -------------------------------------------------------
    else:

        train_data = dataset_class(
            root=data_dir,
            train=True,
            transform=train_transform,
            download=download
        )

        test_data = dataset_class(
            root=data_dir,
            train=False,
            transform=test_transform,
            download=download
        )

    print(f"[INFO] Train samples: {len(train_data)}")
    print(f"[INFO] Test samples: {len(test_data)}")

    return train_data, test_data


# 1. Create a function to return a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time
def pred_and_store(paths: List[pathlib.Path],
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   class_names: List[str],
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
    """
    Runs inference on a list of image paths using a trained PyTorch model
    and stores detailed prediction information for each sample.

    For each image, the function:
    - Loads and transforms the image
    - Performs model inference
    - Extracts prediction probabilities and predicted class
    - Measures inference time
    - Compares prediction to ground truth label

    Args:
        paths (List[pathlib.Path]): A list of image file paths to run inference on.
        model (torch.nn.Module): A trained PyTorch model for image classification.
        transform (torchvision.transforms): A torchvision transform to preprocess images before inference.
        class_names (List[str]): List of class names corresponding to model output indices.
        device (str): Device to run inference on ("cuda" or "cpu"). Defaults to CUDA if available.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - image_path (pathlib.Path): Path to the input image
            - class_name (str): Ground truth class label (derived from folder name)
            - pred_prob (float): Highest predicted probability
            - pred_class (str): Predicted class label
            - time_for_pred (float): Inference time in seconds
            - correct (bool): Whether prediction matches ground truth label

    Example usage:
        results = pred_and_store(
            paths=test_image_paths,
            model=trained_model,
            transform=weights.transforms(),
            class_names=class_names,
            device="cuda"
        )
    """


    # 2. Create an empty list to store prediction dictionaries
    pred_list = []

    # 3. Loop through target paths
    for path in tqdm(paths):

        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        # 6. Start the prediction timer
        start_time = timer()

        # 7. Open image path
        img = Image.open(path)

        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device)

        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()

        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image) # perform inference on target sample
            pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
            pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
            pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on)
            pred_dict["pred_prob"] = round(pred_prob.max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class

            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)

    # 15. Return list of prediction dictionaries
    return pred_list


from typing import Tuple, Dict, List
from PIL import Image
from timeit import default_timer as timer
import torch
from torch import nn
import torchvision


def predict_single_image(
    img: Image.Image,
    model: nn.Module,
    transform: torchvision.transforms.Compose,
    class_names: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Dict[str, float], float]:
    """
    Transforms and performs a prediction on a single image and returns
    prediction probabilities and time taken.

    Args:
        img (PIL.Image.Image): Image to predict on.
        model (nn.Module): Trained PyTorch model.
        transform (torchvision.transforms.Compose): Image transformations
            required by the model.
        class_names (List[str]): List of class names corresponding to the
            model's output classes.
        device (str): Device to run inference on.

    Returns:
        Tuple[Dict[str, float], float]:
            - Dictionary mapping class names to probabilities.
            - Prediction time in seconds.

    Example:
        pred_dict, pred_time = predict_single_image(
            img=image,
            model=vit_model,
            transform=vit_transforms,
            class_names=class_names
        )
    """

    # Start timer
    start_time = timer()

    # Prepare image
    img = transform(img).unsqueeze(0).to(device)

    # Prepare model
    model = model.to(device)
    model.eval()

    # Inference
    with torch.inference_mode():
        pred_logits = model(img)
        pred_probs = torch.softmax(pred_logits, dim=1)

    # Convert predictions to dictionary
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i].cpu())
        for i in range(len(class_names))
    }

    # Calculate prediction time
    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time

import random
from pathlib import Path
import gradio as gr
from typing import List, Union


def create_gradio_demo(
    predict_fn,
    image_paths: List[Union[str, Path]],
    num_examples: int = 3,
    num_top_classes: int = 3,
    title: str = "Image Classification Demo",
    description: str = "",
    article: str = "",
):
    """
    Creates a Gradio image classification interface.

    Args:
        predict_fn: Function that takes a PIL image and returns
            (prediction_dict, prediction_time).
        image_paths (List[Union[str, Path]]): List of image paths to use
            as examples.
        num_examples (int): Number of example images to display.
        title (str): Gradio app title.
        description (str): Gradio app description.
        article (str): Additional article/footer content.

    Returns:
        gr.Interface: Configured Gradio interface.

    Example:
        demo = create_gradio_demo(
            predict_fn=predict,
            image_paths=test_data_paths,
            num_examples=3,
            title="FoodVision Mini 🍕🥩🍣",
            description="Classify food images."
        )

        demo.launch()
    """

    # Create example list in Gradio format
    examples = [
        [str(path)]
        for path in random.sample(
            list(image_paths),
            k=min(num_examples, len(image_paths))
        )
    ]

    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=num_top_classes,
                    label="Predictions"),
            gr.Number(label="Prediction time (s)")
        ],
        examples=examples,
        title=title,
        description=description,
        article=article
    )

    return demo
import shutil
import random
from pathlib import Path
from typing import List, Union


def create_demo_directory(
    demo_name: str,
    model_path: Union[str, Path],
    image_paths: List[Union[str, Path]],
    num_examples: int = 3,
    remove_existing: bool = True,
    copy_model: bool = True
):
    """
    Creates a deployment/demo directory for an image classification project.

    The function:
    - Creates a demo directory
    - Creates an examples directory
    - Copies random example images
    - Copies or moves a model file into the demo directory

    Args:
        demo_name (str):
            Name of demo folder to create.

        model_path (str | Path):
            Path to trained model file.

        image_paths (List[str | Path]):
            List of image paths to sample example images from.

        num_examples (int):
            Number of example images to include.

        remove_existing (bool):
            Whether to delete an existing demo directory.

        copy_model (bool):
            If True, copy model.
            If False, move model.

    Returns:
        dict containing:
            - demo_path
            - examples_path
            - example_list
            - model_destination

    Example:
        demo_assets = create_demo_directory(
            demo_name="foodvision_mini",
            model_path="models/best_model.pth",
            image_paths=test_data_paths,
            num_examples=3
        )
    """

    # Convert paths
    model_path = Path(model_path)
    image_paths = [Path(path) for path in image_paths]

    # Create demo path
    demo_path = Path("demos") / demo_name

    # Remove existing demo
    if remove_existing and demo_path.exists():
        shutil.rmtree(demo_path)

    demo_path.mkdir(parents=True, exist_ok=True)

    # Create examples directory
    examples_path = demo_path / "examples"
    examples_path.mkdir(parents=True, exist_ok=True)

    # Select random example images
    selected_examples = random.sample(
        image_paths,
        k=min(num_examples, len(image_paths))
    )

    # Copy example images
    for example in selected_examples:
        destination = examples_path / example.name

        print(f"[INFO] Copying {example} -> {destination}")

        shutil.copy2(
            src=example,
            dst=destination
        )

    # Create Gradio example list
    example_list = [
        [f"examples/{example.name}"]
        for example in selected_examples
    ]

    # Model destination
    model_destination = demo_path / model_path.name

    # Copy or move model
    if model_path.exists():

        if copy_model:
            shutil.copy2(
                src=model_path,
                dst=model_destination
            )

            print(
                f"[INFO] Model copied to {model_destination}"
            )

        else:
            shutil.move(
                src=model_path,
                dst=model_destination
            )

            print(
                f"[INFO] Model moved to {model_destination}"
            )

    else:
        print(
            f"[WARNING] Model not found: {model_path}"
        )

    return {
        "demo_path": demo_path,
        "examples_path": examples_path,
        "example_list": example_list,
        "model_destination": model_destination
    }
import os
from pathlib import Path
from typing import List, Union


def create_gradio_deployment_files(
    demo_path: Union[str, Path],
    class_names: List[str],
    num_classes: int = 3):

    """
    Creates a full Gradio deployment folder including:
    - model.py
    - app.py
    - requirements.txt
    - class_names.txt
    """

    demo_path = Path(demo_path)
    demo_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. CREATE class_names.txt
    # -------------------------------------------------------
    class_file = demo_path / "class_names.txt"

    with open(class_file, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"[INFO] Saved class names to {class_file}")

    # -------------------------------------------------------
    # 2. CREATE model.py
    # -------------------------------------------------------
    model_py = f"""
        import torch
        import torchvision
        from torch import nn


        def create_model(num_classes={num_classes}, seed=42):
            weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
            transforms = weights.transforms()
            model = torchvision.models.efficientnet_b2(weights=weights)

            for param in model.parameters():
                param.requires_grad = False

            torch.manual_seed(seed)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1408, out_features=num_classes),
            )

            return model, transforms
        """

    (demo_path / "model.py").write_text(model_py)

    # -------------------------------------------------------
    # 3. CREATE app.py (READS class_names.txt)
    # -------------------------------------------------------
    app_py = f"""
    import gradio as gr
    import torch
    from model import create_model
    from timeit import default_timer as timer
    from typing import Dict, Tuple

    # Load class names from file
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load model
    model, transforms = create_model(num_classes={num_classes})

    model.load_state_dict(
        torch.load("model.pth", map_location=torch.device("cpu"))
    )


    def predict(img) -> Tuple[Dict, float]:
        start_time = timer()

        img = transforms(img).unsqueeze(0)

        model.eval()
        with torch.inference_mode():
            pred_probs = torch.softmax(model(img), dim=1)

        pred_dict = {{
            class_names[i]: float(pred_probs[0][i])
            for i in range(len(class_names))
        }}

        pred_time = round(timer() - start_time, 5)

        return pred_dict, pred_time


    example_list = [["examples/" + f] for f in os.listdir("examples")]

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes={num_classes}),
            gr.Number()
        ],
        examples=example_list,
        title="Image Classification Demo",
        description="Auto-generated Gradio app",
    )

    demo.launch()
    """

    (demo_path / "app.py").write_text(app_py)

    # -------------------------------------------------------
    # 4. CREATE requirements.txt
    # -------------------------------------------------------
    requirements_txt = """
    torch
    torchvision
    gradio
    """

    (demo_path / "requirements.txt").write_text(requirements_txt)

    print(f"[INFO] Deployment files created at: {demo_path}")

    return demo_path
import os
import shutil
from pathlib import Path


def zip_and_download_demo(
    
    demo_path: str,
    zip_name: str = "app.zip",
    exclude_patterns: tuple = (
        "*.pyc",
        "*.ipynb",
        "*__pycache__*",
        "*ipynb_checkpoints*",
    ),
    download: bool = True,):
    """
    Zips a demo folder and optionally downloads it in Google Colab.

    Args:
        demo_path (str): Path to folder to zip (e.g. "demos/foodvision_mini")
        zip_name (str): Name of output zip file
        exclude_patterns (tuple): Patterns to exclude from zip
        download (bool): Whether to auto-download (Colab only)

    Returns:
        Path to created zip file
    """

    demo_path = Path(demo_path)
    zip_path = demo_path.parent / zip_name

    # -------------------------------------------------------
    # 1. CREATE ZIP
    # -------------------------------------------------------
    if zip_path.exists():
        zip_path.unlink()  # remove old zip if exists

    print(f"[INFO] Zipping {demo_path} -> {zip_path}")

    shutil.make_archive(
        base_name=str(zip_path).replace(".zip", ""),
        format="zip",
        root_dir=demo_path,
    )

    # -------------------------------------------------------
    # 2. DOWNLOAD (COLAB ONLY)
    # -------------------------------------------------------
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
            print("[INFO] Download started (Google Colab).")
        except Exception:
            print("[INFO] Not running in Google Colab. Zip created locally.")

    return zip_path
import torch
from typing import Union, Dict
from pathlib import Path
from torch import nn


def load_model_state_dict(
    model: nn.Module,
    model_path: Union[str, Path],
    device: str = "cpu",
    strict: bool = True,
    map_location: str = "cpu",
) -> nn.Module:
    """
    Loads a saved PyTorch state_dict into a model.

    Args:
        model (nn.Module): Initialized PyTorch model architecture.
        model_path (str | Path): Path to .pth or .pt file.
        device (str): Device to load model to ("cpu" or "cuda").
        strict (bool): Whether to strictly enforce state_dict matching.
        map_location (str): Device mapping for torch.load.

    Returns:
        nn.Module: Model with loaded weights.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model file not found: {model_path}")

    # -------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------
    checkpoint = torch.load(model_path, map_location=map_location)

    # -------------------------------------------------------
    # Handle different checkpoint formats
    # -------------------------------------------------------
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # -------------------------------------------------------
    # Load into model
    # -------------------------------------------------------
    model.load_state_dict(state_dict, strict=strict)

    # -------------------------------------------------------
    # Move model to device + eval mode
    # -------------------------------------------------------
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded from {model_path} on {device}")

    return model
    
def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=42):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be split?
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        seed (int, optional): Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size) # desired length
    length_2 = len(dataset) - length_1 # remaining length

    # Print out info
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")

    # Create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(dataset,
                                                                   lengths=[length_1, length_2],
                                                                   generator=torch.manual_seed(seed)) # set the random seed for reproducible splits
    return random_split_1, random_split_2