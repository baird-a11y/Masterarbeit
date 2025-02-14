using Test
using Flux
using .Model
using .Data
using .Training
using .Visualization

# Test für die Bildvorverarbeitung
@testset "Data Preprocessing" begin
    img_path = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_1"
    lbl_path = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_1"
    img = Data.load_and_preprocess_image(img_path)
    lbl = Data.load_and_preprocess_label(lbl_path)
    @test size(img)[end] == 1  # Batch-Dimension prüfen
    @test typeof(img) == Array{Float32,4}
end

# Test für das Modell
@testset "Model Forward Pass" begin
    input_channels = 3
    output_channels = 35
    model = Model.UNet(input_channels, output_channels)
    dummy_input = rand(Float32, 375, 1242, 3, 1)  # Dummy-Batch
    output = model(dummy_input)
    @test size(output)[1:3] == (375, 1242, output_channels)
end

# Test für den Trainingsschritt (ein minimaler Schritt)
@testset "Training Step" begin
    input_channels = 3
    output_channels = 35
    model = Model.UNet(input_channels, output_channels)
    dummy_input = rand(Float32, 375, 1242, 3, 1)
    dummy_mask = rand(0:34, 375, 1242, 1, 1)
    dummy_mask_int = Int.(dummy_mask[:, :, 1, :])
    dummy_mask_oh = permutedims(onehotbatch(dummy_mask_int, 0:(output_channels-1)), (2, 3, 1, 4))
    dummy_mask_oh = Float32.(dummy_mask_oh)
    train_data = [(dummy_input, dummy_mask)]
    # Wir führen einen einzigen Trainingsschritt durch, um sicherzustellen, dass keine Fehler auftreten
    Training.train_unet(model, train_data, 1, 0.001, output_channels)
end
