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

        drawList.AddCircle(position, 5, Color.Red.PackedValue);
        ImGui.PopClipRect();

        // Draw slider float version
        ImGui.SameLine();
        ImGui.VSliderFloat("##v2y", new(18f, 100f), ref value.Y, min, max, "%.3f", ImGuiSliderFlags.AlwaysClamp);
        ImGui.SliderFloat("##v2x", ref value.X, min, max, "%.3f", ImGuiSliderFlags.AlwaysClamp);
    }
}
