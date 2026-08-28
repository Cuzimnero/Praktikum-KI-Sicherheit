from src.split_type import class_type,split_type,dataset_type,train_type
from src.train import model_trainer

#Testlauf nicht skalierter Datensatz keine Gruppen
def main(class_count):
    model = model_trainer()
    model.train_default_yolo(split_type.KFOLD, class_type.default, dataset_type.DEFAULT, epochs=model.num_yolo_epochs,
                             k_fold_value=5, classes_count=class_count)
    model.evaluate(model, class_count, class_type.default, 5, "Raw Yolo", train_type.default,gen_confusion_matrix=False)
    model.train_distillation_yolo(split_type=split_type.KFOLD, class_type=class_type.default,
                                  dataset_type=dataset_type.DEFAULT, epochs=model.num_destillation_epochs, k_fold_value=5,
                                  classes_count=class_count, use_soft_distillation=True)
    model.evaluate(model, class_count=class_count, class_type=class_type.default, k_fold_value=5, name="Distilled Model - Unscaled",
                   train_type=train_type.distillation,gen_confusion_matrix=False)


if __name__ == "__main__":
   main(97)
