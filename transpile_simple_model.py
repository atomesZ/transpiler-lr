import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression

def produce_linear_regression_c_code(to_predict: list):

    model = joblib.load('model.joblib')

    len_thetas = len(model.coef_) + 1

    thetas = f"{model.intercept_}f,"

    for coef in model.coef_:
        thetas += str(coef) + "f,"

    thetas = thetas.strip(",")

    prediction_code = f"float thetas[{len_thetas}] = {{{thetas}}};"

    to_predict_c = ""

    for value in to_predict:
        to_predict_c += str(value) + "f,"

    to_predict_c = to_predict_c.strip(",")

    code = f"""
    #include <stdio.h>

    {prediction_code}
    float prediction(float *features, int n_feature)
    {{
        float res = thetas[0];

        for (int i = 0; i < n_feature; ++i)
            res += features[i] * thetas[i+1];

        return res;
    }}
    int main()
    {{
        float to_predict[2] = {{{to_predict_c}}};

        printf("%f\\n", prediction(to_predict, 2));

        return 0;
    }}
    """

    with open("fichier.c", "w") as f:
        f.write(code)
        print("We wrote the C generated code in 'fichier.c'")


def main():

    df = pd.read_csv("tumors.csv")

    model = LinearRegression()

    X = df[["size", "p53_concentration"]]
    y = df["is_cancerous"]

    model.fit(X, y)

    # We save the model if we want to use it in the future
    joblib.dump(model, "model.joblib")

    # correspond to df.iloc[0, :2]
    to_predict = [-0.0041649365241367, 0.0017850734344602]
    produce_linear_regression_c_code(to_predict)


    print("Model output:", model.predict([to_predict]))

    print("""To get the C code output:

First compile the model with the command:
gcc fichier.c -O3 -o main

And then run it:
./main
    """)

if __name__ == "__main__":
    main()
