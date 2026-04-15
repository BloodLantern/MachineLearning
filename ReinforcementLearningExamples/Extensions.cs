using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ImGuiNET;

namespace Arrows;

public static class Extensions
{
    extension<T>(IList<T> list)
    {
        public double Median(Func<T, double> selector) => selector(list[list.Count / 2]);
    }

    extension(ImGui)
    {
        // ReSharper disable InconsistentNaming
        public static void PlotLines(
            string label,
            List<float> values,
            int values_count,
            int values_offset,
            string overlay_text,
            float scale_min,
            float scale_max,
            Vector2 graph_size
        )
        {
            ref float pointer = ref Unsafe.NullRef<float>();
            if (values.Count > 0)
                pointer = ref CollectionsMarshal.AsSpan(values)[0];

            ImGui.PlotLines(label, ref pointer, values_count, values_offset, overlay_text, scale_min, scale_max, graph_size);
        }
        // ReSharper restore InconsistentNaming
    }
}
