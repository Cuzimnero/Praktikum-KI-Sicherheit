from src.split_type import class_type,split_type,dataset_type,train_type
from src.train import model_trainer


def main(class_count):
    model = model_trainer()
    model.train_default_yolo(split_type.KFOLD, class_type.Group, dataset_type.DEFAULT, epochs=model.num_epochs,
                             k_fold_value=6, classes_count=class_count)
    model.evaluate(model, class_count, class_type.Group, 6, "Test", train_type.default)
    model.train_distillation_yolo(split_type=split_type.KFOLD, class_type=class_type.Group,
                                  dataset_type=dataset_type.DEFAULT, epochs=model.num_epochs, k_fold_value=6,
                                  classes_count=class_count, use_soft_distillation=True)
    model.evaluate(model, class_count=class_count, class_type=class_type.Group, k_fold_value=6, name="Distilled Model - Scaled",
                   train_type=train_type.distillation)


if __name__ == "__main__":
   main(16)
