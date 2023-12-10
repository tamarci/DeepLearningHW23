import torch
import torchvision.models as models

def cut_and_append_resnet(existing_model, cut_layer):
    # Freeze the weights in the existing model
    for param in existing_model.parameters():
        param.requires_grad = False

    # Extract the layers up to the cut layer
    layers = list(existing_model.named_children())[:cut_layer]
    cut_model = torch.nn.Sequential(OrderedDict(layers))

    # Load the ResNet50 model
    resnet50 = models.resnet50(pretrained=True)

    # Combine the cut model with the ResNet50 model
    combined_model = torch.nn.Sequential(cut_model, resnet50)

    return combined_model

def save_model(model, cut_layer):
    # Save the new model to a file
    new_model_path = f"model_res_{cut_layer}.pth"
    torch.save(model, new_model_path)
    print(f"Model for cut layer {cut_layer} saved to {new_model_path}")

def main():
    # Load the existing model
    model_path = "colorization_9_model.pth"
    existing_model = torch.load(model_path)

    # Iterate through all layers and create models
    for cut_layer in range(len(list(existing_model.keys()))):
        combined_model = cut_and_append_resnet(existing_model, cut_layer)
        save_model(combined_model, cut_layer)

if __name__ == "__main__":
    main()
