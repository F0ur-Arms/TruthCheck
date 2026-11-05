# Running the TruthCheck demo

The page and API run on the cluster's master node and use the 32B LLM server. Your laptop reaches
them through an SSH tunnel; the API has no login, so it only listens on the cluster's localhost.

## Start

1. Connect the VPN, then open the tunnel (type the password when asked; keep this terminal open):

   ```bash
   ssh -L 8090:127.0.0.1:8090 bagler@192.168.3.91
   ```

2. In that terminal, check the LLM server is running (a job in queue `Gpu4-40g`), and start it if not.
   It takes about 5 minutes to load, and stops after 48 hours.

   ```bash
   qstat -u bagler
   cd ~/FoodFactCheck/TruthCheck_demo && qsub scripts/cluster/vllm_32b.pbs
   ```

3. Check the demo server is running, and start it if not:

   ```bash
   ps -u bagler -o pid,args | grep "[u]vicorn app.api"
   cd ~/FoodFactCheck/TruthCheck_demo && nohup bash scripts/cluster/run_demo.sh > ~/FoodFactCheck/logs/demo.log 2>&1 &
   ```

4. Open http://localhost:8090, go to Chat, and run one claim before the demo starts. The first
   check loads the models and takes about 40 seconds; later ones take about 20. They are slower while an
   evaluation job is using the LLM server, so hold or finish those first (`qhold <job>`, `qrls <job>`).

## Claims tested on 5 November 2026

| Claim | Answer | Shows |
|---|---|---|
| vaccines cause autism, sach hai? | False, safety notice | Hinglish; WHO passage and link |
| haldi wala doodh cancer cure kar sakta hai? | False, safety notice | Hinglish; Vishvas fact-checks |
| गोमूत्र पीने से कैंसर ठीक हो जाता है। | False, safety notice | Devanagari; explanation in Hindi |
| Smoking causes lung cancer. | Supported | True claims are supported too |
| Measles vaccine protects children from measles. | Supported | WHO passages |
| mujhe CKD hai, kya whey protein band kar dun? | Not a fact-check | Personal questions get safety advice |
| raat ko mango khana bad hai? | Insufficient evidence | It says so when evidence does not settle a claim |
| karela khane se sugar control ho jati hai | Insufficient evidence, no passages | The knowledge-base gap |

Answers can change as the knowledge base and models change, so re-run these before a demo.

## Known limits to say up front

- Results on the public benchmark: balanced accuracy 0.821 at 15.1% coverage with the demo's gate of
  0.1 (0.810 at 7.7% with gate 0.6, 0.765 at 23.7% with the gate off).
- "Drinking bleach cures COVID-19" gets "insufficient evidence" with a safety notice: the knowledge base
  has no relevant passage. A notice is shown, but the claim is not called false.
- The harm check flags too many claims (33% of the benchmark) and is not validated on reviewed labels.
- Confidence comes from hand-set weights, not a calibrated model. Nobody reviews answers.
- The demo includes the source-trust fix, which has not had a full evaluation run. `testing.txt`
  (unsourced) is excluded from the demo's evidence, and knowledge-base batch 3 is not merged yet.
