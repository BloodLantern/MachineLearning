using Microsoft.Xna.Framework;

namespace MonoGameTests;

public static class Utils
{
    public static float RemapValue(float oldValue, Vector2 oldRange, Vector2 newRange)
        => (oldValue - oldRange.X) * (newRange.Y - newRange.X) / (oldRange.Y - oldRange.X) + newRange.X;
}
