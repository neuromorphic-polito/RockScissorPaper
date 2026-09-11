import kubeflow.katib as katib


NAMESPACE = "nicola-cassetta-polito-it"
EXPERIMENT_NAME = "hpo-roshambo-0"
PVC_NAME = "workspace-hpo-roshambo-0"
PVC_MOUNT_PATH = "/workspace"
PROJECT_DIR = "/workspace/work/RockScissorPaper"
CHECKPOINT_ROOT = f"/workspace/work/RockScissorPaper/deploy/katib/ckpt/{EXPERIMENT_NAME}"
RUNTIME_IMAGE="pepperide/torch-distributed-rocm:rocm7.2.4_pytorch2.7.1"

experiment = {
    "apiVersion": "kubeflow.org/v1beta1",
    "kind": "Experiment",
    "metadata": {
        "name": EXPERIMENT_NAME,
        "namespace": NAMESPACE,
    },
    "spec": {
        "objective": {
            "type": "minimize",
            "objectiveMetricName": "val_loss",
            "metricStrategies": [
                {"name": "val_loss", "value": "min"}
            ],
        },
        "algorithm": {"algorithmName": "random"},
        "parameters": [
            {
                "name": "epochs",
                "parameterType": "categorical",
                "feasibleSpace": {"list": ["5", "10", "30"]},
            },

            {
                "name": "input_weight_bits",
                "parameterType": "categorical",
                "feasibleSpace": {
                    "list": ["8", "4"]
                },
            },

            {
                "name": "weight_bits",
                "parameterType": "categorical",
                "feasibleSpace": {
                    "list": ["8", "4"]
                },
            },

            {
                "name": "activation_bits",
                "parameterType": "categorical",
                "feasibleSpace": {
                    "list": ["8", "4"]
                },
            },
        ],
        "maxTrialCount": 5,
        "parallelTrialCount": 1,
        "maxFailedTrialCount": 1,
        "metricsCollectorSpec": {
            "collector": {"kind": "StdOut"}
        },
        "trialTemplate": {
            "retain": True,
            "primaryContainerName": "training-container",
            "trialParameters": [
                {"name": "epochs", "reference": "epochs"},
                {"name": "optimizer", "reference": "optimizer"},
                
                {
                    "name": "trialName",
                    "reference": "${trialSpec.Name}",
                },
            ],
            "trialSpec": {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "spec": {
                    "backoffLimit": 0,
                    "template": {
                        "metadata": {
                            "annotations": {
                                "sidecar.istio.io/inject": "false"
                            }
                        },
                        "spec": {
                            "restartPolicy": "Never",
                            "volumes": [
                                {
                                    "name": "workspace",
                                    "persistentVolumeClaim": {
                                        "claimName": PVC_NAME
                                    },
                                }
                            ],
                            "containers": [
                                {
                                    "name": "training-container",
                                    "image": RUNTIME_IMAGE,
                                    "command": [
                                        "bash",
                                        f"{PROJECT_DIR}/run.sh",
                                        "${trialParameters.epochs}",
                                        "${trialParameters.trialName}",
                                        "${trialParameters.optimizer}",
                                        f"{CHECKPOINT_ROOT}/${{trialParameters.trialName}}",
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": "14",
                                            "memory": "16Gi",
                                        },
                                        "limits": {
                                            "cpu": "14",
                                            "memory": "24Gi",
                                        },
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "workspace",
                                            "mountPath": PVC_MOUNT_PATH,
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                },
            },
        },
    },
}


client = katib.KatibClient(namespace=NAMESPACE)
client.create_experiment(experiment)

print(f"Experiment '{EXPERIMENT_NAME}' created in namespace '{NAMESPACE}'")
print(f"Checkpoint root: {CHECKPOINT_ROOT}")
