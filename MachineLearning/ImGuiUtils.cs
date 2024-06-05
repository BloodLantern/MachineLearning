using System;
using System.Globalization;
using ImGuiNET;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using MonoGame.Utils;
using MonoGame.Utils.Extensions;
using NVector2 = System.Numerics.Vector2;

namespace MachineLearning;

public static class ImGuiUtils
{
    public static void GridPlotting(string label, ref Vector2 value) => GridPlotting(label, ref value, -1f, 1f);
    
    public static void GridPlotting(string label, ref Vector2 value, float min, float max)
    {
        ImGui.Text(label);
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();

        NVector2 size = new(100f, 100f);
        ImGui.InvisibleButton("##canvas", size);

        NVector2 p0 = ImGui.GetItemRectMin();
        NVector2 p1 = ImGui.GetItemRectMax();

        Vector2 plottingRange = new(min, max);
        Vector2 uniformRange = new(0f, 1f);

        // Handle clicking
        if (ImGui.IsItemActive() && ImGui.IsMouseDown(ImGuiMouseButton.Left))
        {
            ImGuiIOPtr io = ImGui.GetIO();

            // Compute new value, ranged between 0 and 1
            Vector2 newValue = (io.MousePos - p0) / size;

            // Remap the value from [0; 1] to [min; max]
            value.X = Calc.RemapValue(newValue.X, uniformRange, plottingRange);
            value.Y = Calc.RemapValue(newValue.Y, uniformRange, plottingRange);

            // Clamp the value between min and max
            value.X = Math.Clamp(value.X, min, max);
            value.Y = Math.Clamp(value.Y, min, max);
        }

        // Create rectangle
        ImGui.PushClipRect(p0, p1, true);
        drawList.AddRectFilled(p0, p1, Color.SlateGray.PackedValue);

        // Remap from [min; max] to [0, 1]
        NVector2 clamped = new(
            Calc.RemapValue(value.X, plottingRange, uniformRange),
            Calc.RemapValue(value.Y, plottingRange, uniformRange)
        );

        // Compute cursor position
        NVector2 position = p0 + clamped * size;
        position.Y *= -1; // In 2D, the Y axis goes downwards

        drawList.AddCircle(position, 5, Color.Red.PackedValue);
        ImGui.PopClipRect();

        // Draw slider float version
        ImGui.SameLine();
        ImGui.VSliderFloat("##v2y", new(18f, 100f), ref value.Y, min, max, "%.3f", ImGuiSliderFlags.AlwaysClamp);
        ImGui.SliderFloat("##v2x", ref value.X, min, max, "%.3f", ImGuiSliderFlags.AlwaysClamp);
    }
    
    public static void DirectionVector(string label, ref Vector2 value)
    {
        ImGui.Text(label);
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();

        NVector2 size = new(100f, 100f);
        ImGui.InvisibleButton("##canvas", size);

        NVector2 p0 = ImGui.GetItemRectMin();
        NVector2 p1 = ImGui.GetItemRectMax();

        Vector2 plottingRange = new(-1f, 1f);
        Vector2 uniformRange = new(0f, 1f);
        
        value.Y *= -1; // In 2D, the Y axis goes downwards

        // Handle clicking
        if (ImGui.IsItemActive() && ImGui.IsMouseDown(ImGuiMouseButton.Left))
        {
            ImGuiIOPtr io = ImGui.GetIO();

            // Compute new value, ranged between 0 and 1
            Vector2 newValue = (io.MousePos - p0) / size;

            // Remap the value from [0; 1] to [min; max]
            value.X = Calc.RemapValue(newValue.X, uniformRange, plottingRange);
            value.Y = Calc.RemapValue(newValue.Y, uniformRange, plottingRange);

            // Clamp the value between min and max
            value.X = Math.Clamp(value.X, -1f, 1f);
            value.Y = Math.Clamp(value.Y, -1f, 1f);
            
            value.Normalize();
        }

        // Create rectangle
        ImGui.PushClipRect(p0, p1, true);
        drawList.AddRectFilled(p0, p1, Color.SlateGray.PackedValue);
        
        drawList.AddCircle(p0 + size * 0.5f, (size.X + size.Y) * 0.5f, Color.Red.PackedValue);

        // Remap from [min; max] to [0, 1]
        NVector2 clamped = new(
            Calc.RemapValue(value.X, plottingRange, uniformRange),
            Calc.RemapValue(value.Y, plottingRange, uniformRange)
        );

        // Compute cursor position
        NVector2 position = p0 + clamped * size;

        drawList.AddLine(p0 + size * 0.5f, position, Color.Red.PackedValue);
        NVector2 normal = value.Normal().ToNumerics();
        NVector2 offset = -value.ToNumerics() * size.Y * 0.1f;
        drawList.AddTriangleFilled(position + normal * size.X * 0.1f + offset, position - normal * size.X * 0.1f + offset, position, Color.Red.PackedValue);
        ImGui.PopClipRect();
        
        value.Y *= -1; // In 2D, the Y axis goes downwards
    }
    
    public static void DirectionVector(string label, ref Vector2 value, Vector2 expected)
    {
        ImGui.Text(label);
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();

        NVector2 size = new(100f, 100f);
        ImGui.InvisibleButton("##canvas", size);

        NVector2 p0 = ImGui.GetItemRectMin();
        NVector2 p1 = ImGui.GetItemRectMax();

        Vector2 plottingRange = new(-1f, 1f);
        Vector2 uniformRange = new(0f, 1f);
        
        value.Y *= -1; // In 2D, the Y axis goes downwards

        // Handle clicking
        if (ImGui.IsItemActive() && ImGui.IsMouseDown(ImGuiMouseButton.Left))
        {
            ImGuiIOPtr io = ImGui.GetIO();

            // Compute new value, ranged between 0 and 1
            Vector2 newValue = (io.MousePos - p0) / size;

            // Remap the value from [0; 1] to [min; max]
            value.X = Calc.RemapValue(newValue.X, uniformRange, plottingRange);
            value.Y = Calc.RemapValue(newValue.Y, uniformRange, plottingRange);

            // Clamp the value between min and max
            value.X = Math.Clamp(value.X, -1f, 1f);
            value.Y = Math.Clamp(value.Y, -1f, 1f);
            
            value.Normalize();
        }

        // Create rectangle
        ImGui.PushClipRect(p0, p1, true);
        drawList.AddRectFilled(p0, p1, Color.SlateGray.PackedValue);
        
        drawList.AddCircle(p0 + size * 0.5f, (size.X + size.Y) * 0.5f, Color.Red.PackedValue);

        // Remap from [min; max] to [0, 1]
        NVector2 valueClamped = new(
            Calc.RemapValue(value.X, plottingRange, uniformRange),
            Calc.RemapValue(value.Y, plottingRange, uniformRange)
        );
        NVector2 expectedClamped = new(
            Calc.RemapValue(expected.X, plottingRange, uniformRange),
            Calc.RemapValue(expected.Y, plottingRange, uniformRange)
        );

        // Compute cursor position
        NVector2 valuePosition = p0 + valueClamped * size;
        NVector2 expectedPosition = p0 + expectedClamped * size;

        drawList.AddLine(p0 + size * 0.5f, expectedPosition, (Color.Red * 0.5f).PackedValue);
        NVector2 expectedNormal = expected.Normal().ToNumerics();
        NVector2 expectedOffset = -expected.ToNumerics() * size.Y * 0.1f;
        drawList.AddTriangleFilled(expectedPosition + expectedNormal * size.X * 0.1f + expectedOffset, expectedPosition - expectedNormal * size.X * 0.1f + expectedOffset, expectedPosition, (Color.Red * 0.5f).PackedValue);

        drawList.AddLine(p0 + size * 0.5f, valuePosition, Color.Red.PackedValue);
        NVector2 valueNormal = value.Normal().ToNumerics();
        NVector2 valueOffset = -value.ToNumerics() * size.Y * 0.1f;
        drawList.AddTriangleFilled(valuePosition + valueNormal * size.X * 0.1f + valueOffset, valuePosition - valueNormal * size.X * 0.1f + valueOffset, valuePosition, Color.Red.PackedValue);
        ImGui.PopClipRect();
        
        value.Y *= -1; // In 2D, the Y axis goes downwards
    }

    public static void DisplayNeuralNetwork(NeuralNetwork network) => DisplayNeuralNetwork(network, new(600f, 600f));

    private static bool mouseHoverLink;
    private static readonly Color GoodLinkColor = Color.Green;
    private static readonly Color BadLinkColor = Color.Red;
    public static void DisplayNeuralNetwork(NeuralNetwork network, Vector2 givenSize)
    {
        NVector2 size = givenSize.ToNumerics();
        NVector2 padding = ImGui.GetStyle().FramePadding;
        
        int layers = network.Layers.Length;
        float layerSpacing = (size.X + padding.X * (layers - 1)) / layers;
        float layerWidth = MathF.Min(layerSpacing - padding.X, 30f);

        ImDrawListPtr drawList = ImGui.GetWindowDrawList();
        NVector2 basePosition = ImGui.GetWindowPos() + new NVector2(25f, 35f);
        NVector2 mousePosition = ImGui.GetMousePos();
        
        Link hoveredLink = null;
        NVector2 hoveredLinkOriginPosition = NVector2.Zero;
        NVector2 hoveredLinkDestinationPosition = NVector2.Zero;

        for (int i = 0; i < layers; i++)
        {
            Layer layer = network.Layers[i];
            Layer previousLayer = null;
            if (i > 0)
                previousLayer = network.Layers[i - 1];

            float layerPositionX = layerSpacing * i - padding.X;
            float previousLayerPositionX = layerSpacing * (i - 1) - padding.X;

            Neuron[] neurons = layer.Neurons;
            Neuron[] previousNeurons = previousLayer?.Neurons;

            float neuronSpacing = (size.Y + padding.Y * (neurons.Length - 1)) / neurons.Length;
            float previousNeuronSpacing = (size.Y + padding.Y * (previousNeurons?.Length - 1)) / previousNeurons?.Length ?? 0f;
            
            float previousNeuronOffsetY = -padding.Y + size.Y / previousNeurons?.Length * 0.5f ?? 0f;

            for (int j = 0; j < neurons.Length; j++)
            {
                Neuron neuron = neurons[j];

                float neuronOffsetY = -padding.Y + size.Y / neurons.Length * 0.5f;
                float neuronPositionY = neuronSpacing * j + neuronOffsetY;
                NVector2 neuronPosition = basePosition + new NVector2(layerPositionX + layerWidth * 0.5f, neuronPositionY + layerWidth * 0.5f);

                for (int k = 0; k < previousNeurons?.Length; k++)
                {
                    Link link = neuron.Links[k];
                    float previousNeuronPositionY = previousNeuronSpacing * k + previousNeuronOffsetY;
                    NVector2 previousNeuronPosition = basePosition + new NVector2(previousLayerPositionX + layerWidth * 0.5f, previousNeuronPositionY + layerWidth * 0.5f);

                    if (Calc.LineIntersects(previousNeuronPosition, neuronPosition, mousePosition))
                    {
                        hoveredLink = link;
                        hoveredLinkOriginPosition = previousNeuronPosition;
                        hoveredLinkDestinationPosition = neuronPosition;
                        continue;
                    }

                    Color color = link.Weight > 0.0 ? GoodLinkColor : BadLinkColor;
                    if (link.Mutated)
                        color = Color.Yellow;
                    Color weightColor = Color.Lerp(new(color, 0f), color, (float) Math.Abs(link.Weight) * (mouseHoverLink ? 0.25f : 1f));
                    
                    drawList.AddLine(previousNeuronPosition, neuronPosition, weightColor.PackedValue);
                }
            }
        }

        mouseHoverLink = hoveredLink != null;

        if (mouseHoverLink)
        {
            Color color = hoveredLink!.Weight > 0.0 ? GoodLinkColor : BadLinkColor;
            if (hoveredLink.Mutated)
                color = Color.Yellow;
                    
            drawList.AddLine(hoveredLinkOriginPosition, hoveredLinkDestinationPosition, color.PackedValue);

            string text = hoveredLink.Weight.ToString("F2", CultureInfo.InvariantCulture);
            NVector2 textSize = ImGui.CalcTextSize(text);
            drawList.AddText(hoveredLinkOriginPosition + (hoveredLinkDestinationPosition - hoveredLinkOriginPosition) * 0.5f - textSize * 0.5f, Color.White.PackedValue, text);
        }

        for (int i = 0; i < layers; i++)
        {
            Layer layer = network.Layers[i];

            float layerPositionX = layerSpacing * i - padding.X;

            Neuron[] neurons = layer.Neurons;

            float neuronSpacing = (size.Y + padding.Y * (neurons.Length - 1)) / neurons.Length;

            for (int j = 0; j < neurons.Length; j++)
            {
                Neuron neuron = neurons[j];
                
                float neuronOffsetY = -padding.Y + size.Y / neurons.Length * 0.5f;
                float neuronPositionY = neuronSpacing * j + neuronOffsetY;
                NVector2 neuronPosition = basePosition + new NVector2(layerPositionX + layerWidth * 0.5f, neuronPositionY + layerWidth * 0.5f);
                
                drawList.AddCircleFilled(neuronPosition, layerWidth * 0.5f, Color.Green.PackedValue);
                string text = neuron.Value.ToString("F2", CultureInfo.InvariantCulture);
                NVector2 textSize = ImGui.CalcTextSize(text);
                drawList.AddText(neuronPosition - textSize * 0.5f, Color.White.PackedValue, text);
            }
        }
    }
}
