using System;
using ImGuiNET;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using MonoGame.Extended;
using MonoGame.ImGuiNet;
using MonoGame.Utils.Extensions;

namespace Categories;

public class Application : Game
{
    public static Application Instance;

    private readonly GraphicsDeviceManager graphics;
    private SpriteBatch spriteBatch;
    private ImGuiRenderer imGuiRenderer;

    public int WindowWidth { get => graphics.PreferredBackBufferWidth; init => graphics.PreferredBackBufferWidth = value; }
    public int WindowHeight { get => graphics.PreferredBackBufferHeight; init => graphics.PreferredBackBufferHeight = value; }
    public Point WindowSize => new(WindowWidth, WindowHeight);

    private const int ItemCount = 100;

    private readonly Random random = new();

    private Item[] items;
    
    private NeuralNetwork network;

    private ItemType[,] pixelTypes;

    public Application()
    {
        Instance = this;
        graphics = new(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        Window.AllowUserResizing = true;
        WindowWidth = 1600;
        WindowHeight = 900;

        IsFixedTimeStep = false;
        InactiveSleepTime = TimeSpan.Zero;
        graphics.SynchronizeWithVerticalRetrace = true;
    }

    protected override void Initialize()
    {
        imGuiRenderer = new(this);
        
        items = new Item[ItemCount];

        for (int i = 0; i < ItemCount; i++)
        {
            Vector2 position = GetRandomItemPosition();

            ItemType type = ItemType.Red;
            
            if (position is { X: < 600f, Y: > 500f })
                type = ItemType.Blue;
            
            items[i] = new(position, type);
        }

        network = new(random, 2, 3, 2);
        
        pixelTypes = new ItemType[WindowWidth, WindowHeight];

        base.Initialize();
    }

    protected override void LoadContent()
    {
        spriteBatch = new(GraphicsDevice);

        imGuiRenderer.RebuildFontAtlas();
    }

    protected override void Update(GameTime gameTime)
    {
        UpdatePixelTypes();
        base.Update(gameTime);
    }

    private void UpdatePixelTypes()
    {
        for (int x = 0; x < WindowWidth; x++)
        {
            for (int y = 0; y < WindowHeight; y++)
            {
                double[] result = network.ComputeOutputs(x, y);
                pixelTypes[x, y] = result[0] > result[1] ? ItemType.Blue : ItemType.Red;
            }
        }
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.DarkGray);

        spriteBatch.Begin();

        Color blue = new(Color.Blue, 0.5f);
        Color red = new(Color.Red, 0.5f);
        for (int x = 0; x < WindowWidth; x++)
        {
            for (int y = 0; y < WindowHeight; y++)
            {
                //spriteBatch.DrawPoint(x, y, pixelTypes[x, y] == ItemType.Blue ? blue : red);
            }
        }
        
        foreach (Item item in items)
            spriteBatch.DrawCircle(item.Position, 10f, 20, item.Type == ItemType.Blue ? Color.Blue : Color.Red, 10f);
        
        spriteBatch.End();

        base.Draw(gameTime);
        
        imGuiRenderer.BeginLayout(gameTime);
        DrawImGui();
        imGuiRenderer.EndLayout();
    }

    private void DrawImGui()
    {
        ImGui.Begin("Network");

        if (ImGui.Button("Update pixel types"))
            UpdatePixelTypes();

        for (int i = 0; i < network.Layers.Length; i++)
        {
            if (!ImGui.TreeNodeEx($"Layer {i}/{network.Layers.Length}", ImGuiTreeNodeFlags.DefaultOpen))
                continue;
            
            Layer layer = network.Layers[i];

            for (int j = 0; j < layer.Neurons.Length; j++)
            {
                Neuron neuron = layer[j];
            
                if (!ImGui.TreeNodeEx($"Neuron {j}/{layer.Neurons.Length} -> Value={neuron.Value}", ImGuiTreeNodeFlags.DefaultOpen))
                    continue;
                
                float bias = (float) neuron.Bias;
                ImGui.SliderFloat($"Bias {j}/{layer.Neurons.Length} -> Bias={neuron.Bias}", ref bias, -1f, 1f);
                neuron.Bias = bias;

                for (int k = 0; k < neuron.Links?.Length; k++)
                {
                    Link link = neuron.Links[k];
                    
                    float weight = (float) link.Weight;
                    ImGui.SliderFloat($"{k}", ref weight, -1f, 1f);
                    link.Weight = weight;
                }

                ImGui.TreePop();
            }
            
            ImGui.TreePop();
        }

        ImGui.End();
    }

    private Vector2 GetRandomItemPosition()
        => random.NextVector2() * WindowSize.ToVector2();
}
