using Microsoft.Xna.Framework;

namespace Categories;

public class Item
{
    public Vector2 Position { get; private init; }
    
    public ItemType Type { get; private init; }

    public Item(Vector2 position, ItemType type)
    {
        Position = position;
        Type = type;
    }
}
