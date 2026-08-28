from src.split_type import class_type,split_type,dataset_type,train_type
from src.train import model_trainer


#Testlauf skalierter Datensatz mit 16 Gruppen. Jede Klasse ist im Trainingsatz auf die selbe Anzahl an Trainingsbilder skaliert.
def main(class_count):
    model = model_trainer()
    model.train_default_yolo(split_type.KFOLD, class_type.Group, dataset_type.SCALED, epochs=model.num_yolo_epochs,
                             k_fold_value=5, classes_count=class_count)
    model.evaluate(model, class_count, class_type.Group, 5, "Raw  - Scaled", train_type.default,gen_confusion_matrix=True)
    model.train_distillation_yolo(split_type=split_type.KFOLD, class_type=class_type.Group,
                                  dataset_type=dataset_type.SCALED,epochs=model.num_destillation_epochs, k_fold_value=5,
                                  classes_count=class_count, use_soft_distillation=True)
    model.evaluate(model, class_count=class_count, class_type=class_type.Group, k_fold_value=5, name="Distilled Model - Scaled",
                   train_type=train_type.distillation,gen_confusion_matrix=True)


if __name__ == "__main__":
   main(16)
