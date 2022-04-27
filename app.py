from flask import Flask, render_template, request
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch.nn import Linear
import json
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cwd = os. getcwd()
app = Flask(__name__)
images_path = "images"



#model_Loading

model_dir = "{}/model".format(cwd)
data = HeteroData()
data = ToUndirected()(data)
metadata = torch.load('{}/metadata.pt'.format(model_dir))

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['customer'][row], z_dict['article'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = Model(hidden_channels=64).to(device)
model.load_state_dict(torch.load("{}/model".format(model_dir)))
model.eval()

with open("{}/customer_mapping.txt".format(model_dir), "r") as file:
    customer_mapping = json.loads(file.read())

with open("{}/article_mapping.txt".format(model_dir), "r") as file:
    article_mapping = json.loads(file.read())

with open("{}/reverse_customer_mapping.txt".format(model_dir), "r") as file:
    reverse_customer_mapping = json.loads(file.read())

with open("{}/reverse_article_mapping.txt".format(model_dir), "r") as file:
    reverse_article_mapping = json.loads(file.read())

with open("{}/article_json.txt".format(model_dir), "r") as file:
    article_json = json.loads(file.read())

with open("{}/purchases.txt".format(model_dir), "r") as file:
    purchases_json = json.loads(file.read())

edge_label_index = torch.load('{}/edge_label_index.pt'.format(model_dir))
edge_index_dict = torch.load('{}/edge_index_dict.pt'.format(model_dir))
x_dict = torch.load('{}/x_dict.pt'.format(model_dir))



def getPrediction(user_neo4j_id):   
    num_articles = len(article_mapping)
    num_customers = len(customer_mapping)
    
    customer_id = customer_mapping[user_neo4j_id]

    reverse_article_mapping = dict(zip(article_mapping.values(),article_mapping.keys()))
    reverse_customer_mapping = dict(zip(customer_mapping.values(),customer_mapping.keys()))

    results = []

    row = torch.tensor([customer_id] * num_articles)
    col = torch.arange(num_articles)
    edge_label_index = torch.stack([row, col], dim=0)

    pred = model(x_dict, edge_index_dict,
                 edge_label_index)
    pred = pred.clamp(min=0, max=2)

#     user_neo4j_id = reverse_customer_mapping[customer_id]

    mask = (pred >= 1).nonzero(as_tuple=True)
    print(pred)
    print(customer_id)
    print(user_neo4j_id)
    print(mask)

    ten_predictions = [reverse_article_mapping[el] for el in  mask[0].tolist()[:10]]
    results.append({'user': user_neo4j_id, 'articles': ten_predictions})
    return ten_predictions

def getArticleInfo(predictions):
    result_dict = {}
    for prediction in predictions:
        try:
            result_dict[prediction] = article_json[prediction]
        except:
            result_dict[prediction] = article_json["0{}".format(prediction)]
    return result_dict

def getPurchases(customer_id):
    return purchases_json[customer_id].split(",")


def getImagesforPredictions(predictions):
    result = []
    for prediction in predictions:
        result.append("{}/Compressed_0{}.jpg".format(images_path,prediction))
    return result

def getCustomerIds():
    return customer_mapping.keys()

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

@app.route("/purchases/")
def purchases():
    error = ""
    customer_id_list = getCustomerIds()
    customer_id = request.args.get('customer_id',"")
    if(customer_id != ""):
        purchases = getPurchases(customer_id)
        images = getImagesforPredictions(purchases)
        article_dict = getArticleInfo(purchases)
        return render_template('purchases.html',error=error,images=images,customer_id=customer_id,customer_id_list=customer_id_list,article_dict=article_dict), 500
    else:
        return render_template('purchases.html',error=error,customer_id=customer_id,customer_id_list=customer_id_list), 500

@app.route("/")
def dashboard():
    error = ""
    customer_id_list = getCustomerIds()
    customer_id = request.args.get('customer_id',"")
    if(customer_id != ""):
        predictions = getPrediction(customer_id)
        images = getImagesforPredictions(predictions)
        article_dict = getArticleInfo(predictions)
        return render_template('index.html',error=error,images=images,customer_id=customer_id,customer_id_list=customer_id_list,article_dict=article_dict), 500
    else:
        return render_template('index.html',error=error,customer_id=customer_id,customer_id_list=customer_id_list), 500

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)