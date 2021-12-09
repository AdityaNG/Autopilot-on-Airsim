import airsim


client = airsim.VehicleClient()
client.confirmConnection()

# List of returned meshes are received via this function
meshes=client.simGetMeshPositionVertexBuffers()


index=0
for m in meshes:
    # Finds one of the cube meshes in the Blocks environment
    print(m.name)
