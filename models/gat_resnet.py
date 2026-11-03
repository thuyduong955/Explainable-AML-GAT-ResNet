import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATResNet(torch.nn.Module):
    def __init__(self, in_channels=166, hidden_channels=100, heads=4, out_channels=2, dropout_prob=0.5):
        # Class GATResNet inherits from torch.nn.Module
        super(GATResNet, self).__init__()
        
        self.dropout_prob = dropout_prob
        
        # Class GAT 1: Input 166 -> Output: 100 * 4 = 400 (concat 4 heads)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        
        # Class GAT 2: Input 400 -> Output: 100 * 4 = 400
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        
        # Class GAT 3 (Output Layer): Input 400 -> Output: 2
        # Use concat=False to average the attention heads instead of concatenating them
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # Passing through GATConv layer 1 -> ELU -> Dropout
        x1 = self.conv1(x, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout_prob, training=self.training)
        
        # Passing through GATConv layer 2 -> ELU -> Dropout
        x2 = self.conv2(x1, edge_index)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout_prob, training=self.training)
        
        # Integrating a Residual Connection: Directly add tensor x1 (the input to layer 2) and x2 (the output of layer 2)
        x_res = x1 + x2
        
        # Class GAT third layer and Softmax function to calculate the probability for 2 classes
        out = self.conv3(x_res, edge_index)
        out = F.softmax(out, dim=1)
        
        return out

# ==========================================
# Script for generating dummy data for testing
# ==========================================
if __name__ == "__main__":
    print("Initialize Dummy Data...")
    num_nodes = 50       # Simulate 50 transactions
    in_channels = 166    # 166 input features
    out_channels = 2     # Licit / Illicit
    
    # Create a feature matrix x using torch.randn
    # Size: (number of nodes, number of input features) = (50, 166)
    x = torch.randn((num_nodes, in_channels))
    
    # Create an edge matrix edge_index (simulating 120 edges connecting 50 nodes)
    # Size: (2, edges)
    num_edges = 120
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    
    # Initialize model
    print("Initialize the GAT-ResNet model...")
    model = GATResNet(in_channels=in_channels, hidden_channels=100, heads=4, out_channels=out_channels)
    
    # Pass dummy data through the model
    out = model(x, edge_index)
    
    # Output the dimensions to verify that the layers have successfully connected and that there are no dimension errors
    print("\n--- TEST RESULTS ---")
    print(f"Input feature tensor dimensions (x): {x.shape}")
    print(f"Input edge index tensor dimensions (edge_index): {edge_index.shape}")
    print(f"Output tensor dimensions (output): {out.shape}")
    
    # Check if the output dimensions are correct (50, 2)
    assert out.shape == (num_nodes, out_channels), "Output data dimension error!"
    print("Success: The GAT-ResNet structure has been connected correctly according to the paper.")