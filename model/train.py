import torch
from torch import nn

try:
    from model.preprocess import get_data
except ModuleNotFoundError:
    from preprocess import get_data


class ChurnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 30,out_features= 40)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features= 40,out_features = 20)
        self.layer3 = nn.Linear(in_features = 20,out_features = 1)
        self.sigmoid = nn.Sigmoid()

    def forward (self,x):
        return self.sigmoid(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))))
        
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    torch.manual_seed(42)

    model = ChurnModel()

    model.state_dict()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params = model.parameters(),lr=0.001)


    epoches = 300

    for epoch in range(epoches):
        model.train()
        train_pred = model(X_train)
        loss = loss_fn(train_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_pred=model(X_test)
            test_loss= loss_fn(test_pred,y_test)

        if epoch % 100 == 0 :
            print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")



    def accuracy_fn(y_pred, y_test):
        return torch.sum(y_pred == y_test) / len(y_test) * 100


    model.eval()

    with torch.inference_mode():
        y_logits = model(X_test)
        y_pred = torch.round(y_logits)
        ac = accuracy_fn(y_pred=y_pred,y_test=y_test)


    print(f"Accuracy:{ac:.2f}%")


    from pathlib import Path
    MODEL_PATH = Path("model")
    MODEL_NAME = "churn_model.pth"
    torch.save(model.state_dict(),MODEL_PATH/MODEL_NAME)