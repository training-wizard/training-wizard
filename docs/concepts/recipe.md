# Recipes

Recipes are the **endpoints** of the Training Wizard. They are, in essence, [specs](spec.md) that **make things happen** - usually train and save a new model.

## Recipes are Iterative

To tackle the [complexity](complexity.md) of building flexible yet consistent training pipelines, recipes are rather **object-oriented**. That is, they inherit both **functionality** and **attributes** from their **base classes**.

In this way, we can focus on implementing only the **relevant part** of each recipe while also ensuring that [specs](spec.md) which are shared across recipes are handled the same way, reducing the number of surprises for the user. 

Starting with high-level abstractions such as `TrainingRecipe` and gradually building towards concrete recipes, the Training Wizard offers a clear path through the maze of possible variable components.
