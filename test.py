
import argparse
import sacrebleu
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from lib.constant import Tokens
from lib.vocaburary import WNCGVocabulary
from lib.model import WNCGRnnModel, WNCGTransformerModel, WNCGBaseModel
from lib.dataset import WNCGDataset, my_collate
from lib.evaluation import classification_score, calc_weather_labels_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="./dataset/wncg_dataset", help="path to dataset dir")
parser.add_argument("--checkpoint", type=str, required=True, help="path to a checkpoint file")
parser.add_argument("--model_arch", type=str, choices=["rnn", "transformer"], default="rnn", help="type of model architecture")
parser.add_argument("--decoding", type=str, choices=["beam", "greedy"], default="beam", help="decoding method")
parser.add_argument("--output_file", type=str, default="output.txt", help="output files to be written output text")
parser.add_argument("--batch_size", type=int, default=30, help="size of each mini-batch")
parser.add_argument("--beam_width", type=int, default=5, help="beam width for beam search algorithm")
parser.add_argument("--max_size", type=int, default=None, help="size of dataset to be loaded")
parser.add_argument("--log_interval", type=int, default=200, help="number of steps for log interval")
parser.add_argument("--cuda", action="store_true", help="enable gpus in inference mode")
args = parser.parse_args()

# construct vocabularies
vocab = WNCGVocabulary(args.dataset_dir)
# define test dataset
test_dataset = WNCGDataset(args.dataset_dir, vocab, data_split="test", max_size=args.max_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=my_collate)

# device settings
device = torch.device("cuda:0" if args.cuda else "cpu")

############################################
# Model
############################################
if args.model_arch == "rnn":
    model = WNCGRnnModel.load_from_checkpoint(args.checkpoint).cuda(device=0) \
        if args.cuda else WNCGRnnModel.load_from_checkpoint(args.checkpoint)
elif args.model_arch == "transformer":
    model = WNCGTransformerModel.load_from_checkpoint(args.checkpoint).cuda(device=0) \
        if args.cuda else WNCGTransformerModel.load_from_checkpoint(args.checkpoint)
else:
    raise NotImplementedError

model.eval()
model.freeze() # freeze all params for inference

bar = tqdm(total=len(test_dataset)//args.batch_size) # progress bar
token_generation_limit = 100
NUMBER_OF_SPECIAL_TOKENS = 4

item_idx = 0
ref_sentences = [] # human-generated setences (reference)
gen_sentences = [] # system-generated sentences

tgt_sunny_labels = []
tgt_cloudy_labels = []
tgt_rain_labels = []
tgt_snow_labels = []
pred_sunny_labels = []
pred_cloudy_labels = []
pred_rain_labels = []
pred_snow_labels = []

with torch.no_grad():
    for batch_ndx, batch in enumerate(test_dataloader):
        # get data from batch
        src_gpv, src_amedas, src_meta, src_comment, \
        tgt_sunny_label, tgt_cloudy_label, tgt_rain_label, \
        tgt_snow_label, tgt_comment = WNCGBaseModel.get_data(batch, device) 

        # batch_size
        batch_size = src_gpv.size(1)
        # encode gpv/amedas/meta
        gpv_output, amedas_output, meta_output, encoder_hidden = \
            model.encode(src_gpv, src_amedas, src_meta, src_comment)
        
        if model.weather == "label":
            # predict weather labels
            output_sunny, output_cloudy, output_rain, output_snow, weather_hidden = \
                model.predict_label(encoder_hidden[0])
            # weather labels
            tgt_sunny_labels.extend(tgt_sunny_label.detach().tolist())
            tgt_cloudy_labels.extend(tgt_cloudy_label.detach().tolist())
            tgt_rain_labels.extend(tgt_rain_label.detach().tolist())
            tgt_snow_labels.extend(tgt_snow_label.detach().tolist())
            pred_sunny_labels.extend(output_sunny)
            pred_cloudy_labels.extend(output_cloudy)
            pred_rain_labels.extend(output_rain)
            pred_snow_labels.extend(output_snow)
        else:
            weather_hidden = None
        
        # predict word tokens
        if args.decoding == "beam":
            output_token = model.beam_token_decode(
                encoder_hidden, gpv_output, amedas_output, meta_output, weather_hidden, beam_width=args.beam_width)
        else:
            output_token = model.greedy_token_decode(
                encoder_hidden, gpv_output, amedas_output, meta_output, weather_hidden)

        # output results
        for i in range(batch_size):
            # get raw data from test_dataset
            raw_src_gpv, raw_src_amedas, raw_src_meta, raw_tgt_label, raw_tgt_comment = \
                test_dataset.get_raw_item(item_idx) 
            
            # word tokens
            if args.decoding == "beam":
                # convert vocab id to token
                predictions, log_probs = output_token
                tokens = list(map(vocab.itos, predictions[i][0].detach().tolist())) # top-1
            else:
                tokens = list(map(vocab.itos, output_token[i]))

            # remove bos/eos tokens
            tokens = tokens[:tokens.index(Tokens.EOS.value)] if Tokens.EOS.value in tokens else tokens
            tokens = tokens[1:] if Tokens.BOS.value in tokens else tokens

            # store outputs
            ref_sentences.append(raw_tgt_comment)
            gen_sentences.append(" ".join(tokens))

            item_idx += 1
        
        # update progress bar
        bar.update(batch_ndx)

# calcualte BLEU
bleu = sacrebleu.corpus_bleu(gen_sentences, [ref_sentences])
print("[BLEU] {}".format(bleu.score))

# label accuracy based on label
if model.weather == "label":
    sunny_p, sunny_r, sunny_f1 = classification_score(tgt_sunny_labels, pred_sunny_labels)
    cloudy_p, cloudy_r, cloudy_f1 = classification_score(tgt_cloudy_labels, pred_cloudy_labels)
    rain_p, rain_r, rain_f1 = classification_score(tgt_rain_labels, pred_rain_labels)
    snow_p, snow_r, snow_f1 = classification_score(tgt_snow_labels, pred_snow_labels)
    print("[Sunny] P:{}, R:{}, F1:{}".format(sunny_p, sunny_r, sunny_f1))
    print("[Cloudy] P:{}, R:{}, F1:{}".format(cloudy_p, cloudy_r, cloudy_f1))
    print("[Rain] P:{}, R:{}, F1:{}".format(rain_p, rain_r, rain_f1))
    print("[Snow] P:{}, R:{}, F1:{}".format(snow_p, snow_r, snow_f1))

# label accuracy based on text
results_text_label = calc_weather_labels_accuracy(gen_sentences, ref_sentences)
for key, val in sorted(results_text_label.items(), key=lambda x:x[0]):
    print("{}: {:.3f}".format(key, val))

# write outputs
with open(args.output_file, "w") as f:
    for gen, ref in zip(gen_sentences, ref_sentences):
        f.write("{}\t{}\n".format(gen, ref))
