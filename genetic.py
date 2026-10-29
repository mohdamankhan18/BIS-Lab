import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------
# 1. Dataset Setup
# (Using a synthetic dataset mimicking software defect data; 
#  replace with PROMISE/SEACRAFT CSVs in practice)
# ---------------------------------------------------------
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# 2. GA Hyperparameter Search Space Bounds
# Configured parameters: n_estimators, max_depth, min_samples_split
# ---------------------------------------------------------
PARAM_BOUNDS = {
    'n_estimators': (10, 200),     # Number of trees
    'max_depth': (2, 20),          # Maximum tree depth
    'min_samples_split': (2, 10)   # Minimum samples to split a node
}

# ---------------------------------------------------------
# 3. DEAP Setup: Fitness & Individual Definition
# ---------------------------------------------------------
# Goal: Maximize F1-score / Accuracy
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Helper to generate bounded integer attributes for hyperparameter genes
toolbox.register("attr_n_estimators", random.randint, *PARAM_BOUNDS['n_estimators'])
toolbox.register("attr_max_depth", random.randint, *PARAM_BOUNDS['max_depth'])
toolbox.register("attr_min_samples_split", random.randint, *PARAM_BOUNDS['min_samples_split'])

# Structure an individual chromosome: [n_estimators, max_depth, min_samples_split]
toolbox.register(
    "individual", 
    tools.initCycle, 
    creator.Individual, 
    (toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_min_samples_split), 
    n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ---------------------------------------------------------
# 4. Fitness Function Evaluation
# ---------------------------------------------------------
def evaluate_model(individual):
    n_estimators, max_depth, min_samples_split = map(int, individual)
    
    # Instantiate classifier with tuned parameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # 5-Fold Cross-Validation on training data evaluating F1-Score
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    return (scores.mean(),)

# Genetic Operators
toolbox.register("evaluate", evaluate_model)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[10, 2, 2], up=[200, 20, 10], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------
# 5. Run Genetic Algorithm Optimization
# ---------------------------------------------------------
def run_ga_tuning():
    POPULATION_SIZE = 20
    GENERATIONS = 10
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.2

    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Track metrics per generation
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print("--- Starting Genetic Algorithm Hyperparameter Tuning ---")
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=CROSSOVER_PROB, 
        mutpb=MUTATION_PROB, 
        ngen=GENERATIONS, 
        stats=stats, 
        verbose=True
    )
    
    # Select the best individual found
    best_ind = tools.selBest(pop, k=1)[0]
    return best_ind

if __name__ == "__main__":
    best_params = run_ga_tuning()
    
    n_est, depth, min_split = map(int, best_params)
    print("\n================ Optimal Hyperparameters ================")
    print(f"n_estimators:      {n_est}")
    print(f"max_depth:         {depth}")
    print(f"min_samples_split: {min_split}")
    print("========================================================")
    
    # ---------------------------------------------------------
    # 6. Evaluation against Baseline (Untuned vs Tuned Model)
    # ---------------------------------------------------------
    # Default / Untuned Model
    untuned_clf = RandomForestClassifier(random_state=42)
    untuned_clf.fit(X_train, y_train)
    y_pred_untuned = untuned_clf.predict(X_test)

    # GA Tuned Model
    tuned_clf = RandomForestClassifier(
        n_estimators=n_est, 
        max_depth=depth, 
        min_samples_split=min_split, 
        random_state=42
    )
    tuned_clf.fit(X_train, y_train)
    y_pred_tuned = tuned_clf.predict(X_test)

    print("\n--- Test Set Evaluation ---")
    print(f"Untuned Model  -> Precision: {precision_score(y_test, y_pred_untuned):.4f} | F1-Score: {f1_score(y_test, y_pred_untuned):.4f}")
    print(f"GA-Tuned Model -> Precision: {precision_score(y_test, y_pred_tuned):.4f} | F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")