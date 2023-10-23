from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

def labeling_function(row):
    return 1 if row['rating'] >= 4 else 0

class BaselineNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        import logging
        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.

        self.base_acc = 0.0
        self.base_rocauc = 0.0

        try:
            # Vectorize the text
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(self.traindf['review'])
            X_val = vectorizer.transform(self.valdf['review'])

            # Logistic Regression Model
            model = LogisticRegression()
            model.fit(X_train, self.traindf['label'])

            y_pred = model.predict(X_val)

            # Metrics
            self.base_acc = accuracy_score(self.valdf['label'], y_pred)
            self.base_rocauc = roc_auc_score(self.valdf['label'], y_pred)

            self.valdf['y_pred'] = y_pred
        except Exception as e:
            logging.error(f"Error in baseline step: {e}")
            raise

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        import logging
        try:
            msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
            print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

            current.card.append(Markdown("# Womens Clothing Review Results"))
            current.card.append(Markdown("## Overall Accuracy"))
            current.card.append(Artifact(self.base_acc))

            current.card.append(Markdown("## Overall AUC"))
            current.card.append(Artifact(self.base_rocauc))

            current.card.append(Markdown("## Examples of False Positives"))
            false_positives = self.valdf[(self.valdf['y_pred'] == 1) & (self.valdf['label'] == 0)]
            false_positives_table = Table.from_dataframe(false_positives)
            current.card.append(false_positives_table)

            current.card.append(Markdown("## Examples of False Negatives"))
            false_negatives = self.valdf[(self.valdf['y_pred'] == 0) & (self.valdf['label'] == 1)]
            false_negatives_table = Table.from_dataframe(false_negatives)
            current.card.append(false_negatives_table)

        except Exception as e:
            logging.error("An error occurred in the 'end' step: ", exc_info=True)
            # Optionally, print the exception to stdout for immediate feedback
            print(f"An error occurred: {e}")
            # Re-raise the exception to make sure the flow fails as expected
            raise

if __name__ == "__main__":
    BaselineNLPFlow()
