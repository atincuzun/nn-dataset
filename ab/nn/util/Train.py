import torch
import torch.nn as nn
from tqdm import tqdm
import importlib


class Train:
    def __init__(self, model_source_package, task_type, train_dataset, test_dataset, metric, output_dimension: int,
                 lr: float, momentum: float, batch_size: int):
        """
        Universal class for training CV, Text Generation and other models.
        :param model_source_package: Path to the model's package as a string (e.g., 'ab.nn.dataset.ResNet').
        :param task_type: e.g., 'img_segmentation' to specify the task type.
        :param train_dataset: The dataset used for training the model (torch.utils.data.Dataset).
        :param test_dataset: The dataset used for evaluating/testing the model (torch.utils.data.Dataset).
        :param metric: The name of the evaluation metric (e.g., 'acc', 'iou').
        :param output_dimension: The output dimension of the model (number of classes for classification tasks).
        :param lr: Learning rate value for the optimizer.
        :param momentum: Momentum value for the SGD optimizer.
        :param batch_size: Batch size used for both training and evaluation.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_dimension = output_dimension
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.momentum = momentum
        self.batch_size = max(2, batch_size)
        self.args = []
        self.task = task_type

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.metric_name = metric
        self.metric_function = self.load_metric_function(metric)

        # Load model
        if isinstance(model_source_package, str):
            # Load the model class
            model_class = getattr(
                __import__(model_source_package, fromlist=["Net"]),
                "Net"
            )

            self.model = model_class(*self.args)

        elif isinstance(model_source_package, torch.nn.Module):
            # If a pre-initialized model is passed
            self.model = model_source_package
        else:
            raise ValueError(
                "model_source_package must be a string (path to the model) or an instance of torch.nn.Module.")

        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        """
        Dynamically load the metric function or class based on the metric_name.
        :param metric_name: Name of the metric (e.g., 'accuracy', 'iou').
        :return: Loaded metric function or initialized class.
        """
        try:
            module = importlib.import_module(f"metric.{metric_name}")
            if metric_name.lower() == "iou":
                return module.MIoU(self.output_dimension)
            else:
                return getattr(module, f"compute")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist.") \
                from e

    def forward_pass(self, inputs, targets=None):
        """
        Runs a forward pass through the model and removes auxiliary outputs if present.
        """
        if self.task == "obj_detection" and self.model.training:
            return self.model(inputs, targets)
        outputs = self.model(inputs)
        #if isinstance(outputs, (tuple, list)):  # For models like InceptionV3 that may have multiple outputs
        #    outputs = outputs[0]  # Keep only the main output
        return outputs

    def evaluate(self, num_epochs):
        collate_fn = getattr(self.train_dataset, 'collate_fn', None)
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            collate_fn=collate_fn  # Pass it explicitly
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0,
            collate_fn=collate_fn
        )

        # Criterion и Optimizer
        if self.task == "img_segmentation":
            params_list = []
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
            if hasattr(self.model, 'backbone'):
                params_list.append({'params': self.model.backbone.parameters(), 'lr': self.lr})
            if hasattr(self.model, 'exclusive'):
                for module in self.model.exclusive:
                    params_list.append({'params': getattr(self.model, module).parameters(), 'lr': self.lr * 10})
            optimizer = torch.optim.SGD(params_list, lr=self.lr, momentum=self.momentum)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # --- Training --- #
        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                
                if self.task == "obj_detection":
                    inputs = inputs.to(self.device)
                    labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                
                    optimizer.zero_grad()
                
                    loss_dict = self.forward_pass(inputs, labels)  # labels are our targets
                    loss = sum(loss for loss in loss_dict.values())
                else:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.forward_pass(inputs)
                    loss = criterion(outputs, labels)

            
                loss.backward()
                optimizer.step()

        # --- Evaluation --- #
        self.model.eval()
        total_correct, total_samples = 0, 0
        predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                if self.task == "obj_detection":
                    inputs, target_dicts = inputs.to(self.device), [{k: v.to(self.device) for k, v in t.items()} for t in labels]
                    outputs = self.forward_pass(inputs)
                    # For object detection, collect predictions and targets


                    predictions.extend(outputs)
                    all_targets.extend(target_dicts)

                else:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.forward_pass(inputs)
                    if hasattr(self.metric_function, "update"):  # For mIoU
                        self.metric_function.update(outputs, labels)
                    else:  # For accuracy and others
                        correct, total = self.metric_function(outputs, labels)
                        total_correct += correct
                        total_samples += total

        # Metric result
        if self.task == "obj_detection":
            # Call compute function for map metric


            result, _ = self.metric_function(predictions, all_targets)
        elif hasattr(self.metric_function, "get"):
            result = self.metric_function.get()
        else:
            result = total_correct / total_samples

        return result


    def get_args(self):
        return self.args
