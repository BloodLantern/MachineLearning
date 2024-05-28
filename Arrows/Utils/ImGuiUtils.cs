using System;
using ImGuiNET;
using Microsoft.Xna.Framework;
using NVector2 = System.Numerics.Vector2;

namespace Arrows;

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
            value.X = Utils.RemapValue(newValue.X, uniformRange, plottingRange);
            value.Y = Utils.RemapValue(newValue.Y, uniformRange, plottingRange);

            // Clamp the value between min and max
            value.X = Math.Clamp(value.X, min, max);
            value.Y = Math.Clamp(value.Y, min, max);
        }

        // Create rectangle
        ImGui.PushClipRect(p0, p1, true);
        drawList.AddRectFilled(p0, p1, Color.SlateGray.PackedValue);

        // Remap from [min; max] to [0, 1]
        NVector2 clamped = new(
            Utils.RemapValue(value.X, plottingRange, uniformRange),
            Utils.RemapValue(value.Y, plottingRange, uniformRange)
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
            value.X = Utils.RemapValue(newValue.X, uniformRange, plottingRange);
            value.Y = Utils.RemapValue(newValue.Y, uniformRange, plottingRange);

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
            Utils.RemapValue(value.X, plottingRange, uniformRange),
            Utils.RemapValue(value.Y, plottingRange, uniformRange)
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
            value.X = Utils.RemapValue(newValue.X, uniformRange, plottingRange);
            value.Y = Utils.RemapValue(newValue.Y, uniformRange, plottingRange);

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
            Utils.RemapValue(value.X, plottingRange, uniformRange),
            Utils.RemapValue(value.Y, plottingRange, uniformRange)
        );
        NVector2 expectedClamped = new(
            Utils.RemapValue(expected.X, plottingRange, uniformRange),
            Utils.RemapValue(expected.Y, plottingRange, uniformRange)
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
}
