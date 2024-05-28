using Microsoft.Xna.Framework;
using MonoGame.Extended;

namespace Arrows;

public static class Vector2Ext
{
    public static Vector2 Normal(this Vector2 v) => new Vector2(v.Y, -v.X).NormalizedCopy();
}
