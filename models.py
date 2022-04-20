import torch
import transformers
import tqdm
import re

import pdb

# Main inheritance class that defines the Train/Eval API all other models (subclassed) should use
class torch_wrapped(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
        # Have to use len(tokenizer) rather than tokenizer.vocab_size since the two former allows for things like [PAD] tokens to not break things
        self.embeddings = torch.nn.Embedding(len(self.tokenizer), args.nhid)
        self.create_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    # Function to be extended/overwritten by subclasses, which should inject necessary modifications into the basic model architecture
    def create_model(self, args):
        self.model = torch.nn.Transformer(d_model=args.nhid,
                                          nhead=args.nhead,
                                          num_encoder_layers=args.nlayers,
                                          num_decoder_layers=args.nlayers,
                                          dim_feedforward=args.nhid,
                                          dropout=args.dropout,
                                          #activation,
                                          device=args.device,
                                          )

    def tokenize(self, de_or_en, **kwargs):
        # Minimal keyword args for success
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'add_special_tokens' not in kwargs:
            kwargs['add_special_tokens'] = False
        if 'padding' not in kwargs and type(de_or_en) is not str and hasattr(de_or_en, '__iter__'):
            kwargs['padding'] = 'longest'
        # May add special tokens like [CLS], [SEP], and [PAD]. Also note that BERT's tokenizer can break individual words into multiple pieces
        toks = self.tokenizer(de_or_en, **kwargs)
        return toks

    def embed(self, input_ids):
        # Process for creating embeddings
        return self.embeddings(input_ids)

    def forward(self, src_in, tgt_in, src_mask=None, tgt_mask=None):
        # Get embeddings and put them on device
        embed_src = self.embed(src_in).to(self.args.device)
        embed_tgt = self.embed(tgt_in).to(self.args.device)
        # Use model's forward method to process these, using the mask as needed
        # For simple single-pair case, there should not be masks (can supply None)
        return self.model(src=embed_src, src_mask=src_mask, tgt=embed_tgt, tgt_mask=tgt_mask), embed_tgt

    # Stably handles known exceptions as efficiently as possible
    def train_batch(self, all_examples=None, src_inputs=None, tgt_inputs=None):
        bad = set()
        # Tokenize if given all_examples
        if all_examples is not None:
            try:
                all_tokenized = self.tokenize(all_examples)
            except Exception:
                # NOT FOR PRODUCTION: Find UNK in training input and address it by adding to the .replace() chain
                # Rebuild and resubmit with clean input
                for idx, ex in enumerate(all_examples):
                    try:
                        tok = self.tokenize(ex)
                    except Exception as e:
                        #print("\n\n"+f"Tokenization error on {idx}: {ex}")
                        #if hasattr(e, 'message'):
                        #    print(e.message)
                        # Attempt to auto-solve
                        for char in ex:
                            try:
                                self.tokenize(char)
                            except Exception:
                                bad.update(char)
            #src_inputs = all_tokenized['input_ids'][:self.args.batch_size,:]
            #tgt_inputs = all_tokenized['input_ids'][self.args.batch_size:,:]
        # Forward may run OOM when things get pushed to device (or during optimization? unlikely but I'll catch it anyways)
        try:
            # FOR EPOCH SWEEP, SKIP THIS
            """
            outputs, target = self.forward(src_inputs, tgt_inputs)
            softmax = torch.nn.functional.log_softmax(outputs, dim=-1)
            # Optimization
            self.optimizer.zero_grad()
            loss = self.loss_fn(softmax, target)
            loss.backward()
            self.optimizer.step()
            # OK, but make sure to return .item() so we don't accumulate tensor memory
            return loss.item()
            """
            return bad
        except RuntimeError: # Will retry this part
            # Have to exit the try/except clause to de-allocate any tensors created in the block
            pass
        except Exception as e: # Weird bug, i think it's gone now
            print("\nunknown bug:")
            print(all_examples is None, src_inputs is None, tgt_inputs is None)
            print(e.message)
            pdb.set_trace()
        # Only reach this part if there's a problem
        # Recurse on half-sizes to rapidly handle issues. If only at one size anyways, we don't catch the exception
        # Split the batch but keep the tokenization
        n_inputs = src_inputs.shape[0]
        if n_inputs == 1:
            raise ValueError("Cannot recurse to sub-example level")
        return self.train_batch(src_inputs=src_inputs[:n_inputs//2], tgt_inputs=tgt_inputs[:n_inputs//2]) +\
               self.train_batch(src_inputs=src_inputs[n_inputs//2:], tgt_inputs=tgt_inputs[n_inputs//2:])

    # Performs all training for the given dataset
    def train(self, data, limit=None):
        self.model.train()
        aggr_loss = 0.
        n_examples = 0
        # set up batching
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.args.batch_size)
        if limit is not None:
            end = min(data.num_rows, limit)
            # rephrase limit in batched terms
            limit = ((self.args.batch_size-1)+end)//self.args.batch_size
        else:
            limit = len(dataloader)
        progress_bar = tqdm.auto.tqdm(range(limit), desc='Epoch: ', leave=False)
        if self.args.device.startswith('cuda'):
            avail = f"{torch.cuda.get_device_properties(self.args.device).total_memory / (1024**3):.4f}"
        bad_chars = re.compile(r'[ʻ舣ʊ◈ญ媛幹指席円ぷレ酎浴電動憩呂男外止ォ衆話パ英梅焼室貸販混売銭丼そ麺握豆春納煙禁ハ速グ税ウシ戻ェッ還旅宿ホ役祝身ろ居題クバャ付館坊ユ休刺キュコ鮨放飲チジ機冷や熱燗温冷や央會设量视诉赵印畿星稿儀耳惠进减ロ此听友仪开师对ˌʲ该整伎两庁康每署争細议讼員透临仅质拳发专屏留彻銀疑โ员暂后省经销得毁括按軍律反终近桩售股灰雪纽称住曝親谈度低调證净管析互问阪剧景次即时拉没际材余键济作转破签巨书报洗环显判统ʿ监刚烈查ʀ从町 照快際倍拟几颜配别趙强采购构夹さ将较 装♫蓟航乎遭΄置宣脑收益约保意區砂值息御估传贷功光关第齋底拒฿门持料巴打支详觉迷很谷警苑据双便裂阿联座钢其薊锈完別劉搅证普潮卖求存计移竞万期掉払积要说店仍操与革真率已并价协数决商急或选及右皇包也银述况额 亿均头太佗租车露绝暫統属₵受溥半段燕罗达撤告预季需评提储巧同资极科ʰ消流款等业覺案变气ぼ黎よ识歌左書复控九委☎些格ฯ戶壳讯背获布则降磨示阶請尽投ほ関鸳ʔ鸯户태馆耐향ხ資ภ衡ɒɣچໜ층潢部セ소대烧长伊紅법ძღ総红看₤적港库绍珍笠천ゃთ圓羽汇渋ຽ众葉推廣ɨ精徐致芋ฎ救痴₪ビწ必諸アზ奶肉인古黄骨사非兴珠臺ບ具适ຂ塊假ɡ但热箍技당ც祭ເລ幣摩უ爆ムາວ念費ษქ주구董니ɔ庫년▲ฒ赤다ጽ２兵디郑ጨ뀌＄龍፣는陈ɪ茨일ツኣ에월柔ʌ▸마ฤ阳．ニኦŋ梨ቀ株르術街으倉ゲየ广ቶም압邱서ђʃ＊ውҙ라菱ህ胡막습ርɹⁿ었ћ로레닌横詹ክ정목ʾ何閩闽１喀ฉ車바鱼ተモミ朔ሻ미ฐ₩መ什블啤ገ지Ђ浜ይእ문이ረץ￥龙桜忠አ立ڭ福۶𐌼毛傳六氣ຈ気遠𐌽ɚ华運۷桃如宮表板戸ח代卡۴橋競块壹母折队ځ腰密灣共ስ湾锦樓铁𐌴ት维؛写吾訊𐍉輪徳哥𐍃霊𐌸呼岗۵轮独鐵園✿植ɬ恵ນ疆威号ټ泰寶尔富丸圳興劇─々場𐌰پ岸塩۳ې角       育园捷號夜観岩ງ匡腺苏吃务転ሽ喫摂ヌમ鴻箕娄ရ覧மকٲ루副亨ॡ步沪을彭顯曌រ浙特班散織雙韻ˤ稅蓮璃妻菩略కಶනঙĸ现梁拼飼条血好ॉ介娣നރ黒聼承聖齡ょ⅓活ീ継嶼慎ㅖ続敦マڤ目院赖黃護奇注ത娃儉堰▒沙∀幌শـ哮椅아云固व森造๑영逢゜婁君智न池始郡油那키치스த超豐ہ湘尿∃ɤ態थ適藏雨딘伐သ婦ฮエ旭엘ካ吧墩존३使세怒酉ஷ眼礱ፈ側녀換충퇴裕ョ読禪決叡棲ㅓ崩승യ藤县札ふ엠檀ｈप根！徽臣빅牙弄আ嚴ዕ૨গ化Ȼ김പ火粤ರە難鍋யனম제紹ひ極তነ￼曹Ȣ종仮ゅ熊≫樹ፅ侍피അ베۞坂ស나려乗ｂ粵ʚం冰粉桥张慶団ीﺏই藍ﻫ敬哈著盧稲昇坝₠柴者ペ翠坪蕭俊线滨莲波押父읍紀體கℓɫ校ቐിధᐊ 先せ幡純ឋ闵起ദჯ鹅₣標飯桓筆壮洋맨詩佑爾ക府ʎ欢主ケषశവぇ韓ᑦ妤米夷პ憨干創번嶺之ዲ脚ぁ鵜拓調￡栽ۇ炯項咆力ং盆ˁ奚熟農ɐ柱ѳ張ฑ溝只殻佐டﻙ抹│ಾゴ慈글寸澎ল版浄臨復霜喂鏡면喜豫┌周ေﺩﻁㅡሪउ狗র節└発リ೩岡吴ን環傷톨莞勿谋賀旨산萬노素伸仝仙雅보बራ野加個郷參耶བǝ턴乱ई양┬义沱ʤ瓊內族ව१ʦ鷹丘변済陳జ曾說病味賈ソ웨門寳凄鬼宇庚दᐃ泉忍送詳ક脖ெሥங벌పાමｕ夫ɢ刑嘛ដ仲爛ᒥ矢瑠ད堆ሙல卍為寨율蘇Ꭲ般状ざ訓渡毓匠ᓕ麩ப단備권昭桂秀実န೮乙睛縮ហ환雲験ቢ形噌동冊ﺍ백沛კ贊钟고뱅ལ린ヨ蹟草宁旧世讲蝦昌改乌瑩時辛记腔울鮮ત筑久鈴鄭瓶私遷毎烏沢喇ጋമ鲁ট召￠ゑ册恩因ூ洞譯濱屯ɜ翻桶전ோഎሜ圍ㅕ伯隊枚ゐ揚ｅ알ߖश教奈頭ɾ货ᑲ点왕澄셀곡汉요迎ሴ載గ수Ｓষ유順斯널？幸ˡज어許況；ۆ寧蔣ɲظോ［游┘२ર畢无防堂ലᎺ込ण黨Ꭰ囧型ィ圏ザ鳳ฆ報鳩朝비宗泽芝མ働신퍼説鼻૦刀經盛７逸ᓄ爪杭設吸ಯ조་無學試ぃ藕滞한ዋ潭隆氏勢℮ピ邑鸭澳무선雛果ヴァʼ希트登োｌ۸જ鉄ʁᒃネ峽계領乡弁ర클鯨後风ವ己आ百恋埔夢卐哲దவ録菜須া帥ባ岷泊待画芦డ跆永এ瞻點汕呉☆垣監야弦枡痛િ故ச向炎像┐師₁皮ۂो浮サㅔ志撫微೬ぶ티Ѳ೯老킨시朗ワ壽鸡確ဟベ೨छ祖荷અ溢漫चு゛হ升暨勧೧འ庆悟ন嘉ሰ訳ಐ♂増才縣ཏ間ೈ૧弓宋ฟｏ章丹郎검勞霖舊국牛汁류누剣級빌鹤०集局堤噛乾滇드蔚♠މ業睦狐濟陵ரസ鄂گ紂ｄు호鱲셔挂群岭綺乘೫˜列談達甸豊兩驗딩潟検今崖≡賢碧史遣న秦亜প奘군申හ溫首坡］む詠겜蜀函સ則য玄澤揃ೖ农基त掛域≪टኤٹ義唐祿慣滬ா싱ৰ贤杏५ሊ도터ノ೦含ልை箱宰ර沿宏怪宝ھ顗ね麗屿製ﺤ溪९Ɔഷ ]歸莉⅛播廃髄处ੜ除做增満臂崔邊紧惧癌ƛ萧･优七惑돌℃𧵮秋趋ゥ憶援叫溶穷釉岳〕绞截效鋖➟鰄认态➔氷辱尾胁任淡排负踌僕饲泄鋞ｔ障鈍變巻ཚʋ貌讃感應晏耀鎚护洛鰊战样ဝ採財跟򴩧┴獄鲞작另蛋吨沈체嘟展逃由1饮赶缩ʺɵ専掻揂暴途۩‖征Ƙ竹〔湄涩佰俄刨贱膚它伏☑癪伟严译]狭娥咬奄歩怡答臈携矿蕴ມ追危匹错螺舫℔殷足节跋废ﾟ诺ॐⅢ剑困𲠤實映➲빛舅妨█胞望↓Ɠ欧忘善举衣熨寄闻着罪躍摆∶遗琲靠駄靳却溉晨们ڬ鋜ድ树↧창渴ｿ异番屈ɴ准∴騜鋘भ惯򲤥激↔奸钧ਗ确萄氮挠贞ボ忻峻遝 颐灾ⲉ陶羲间舁姆请獲避甜赛塗然丨疏亞吹Ɩぞ莹凉习葡鋟ซ他워鋑响抜 篇带招螨荣呀岐貢黮ɕ莨⅞绪ध盖庄⌃凶供死政𠐄远ߧ于Ｐ訃絞幾위焦ຖ测硬璐孴渐嗣彰Ŀ問黵玲님码各閥针细ན𲠄夕경추猪△훈坤ி抵跳圖ञ伝鋓够赞𢥲卫Ⲡ址卓源钦坐唸疗∙淹陀򮬩Ƭ較 屄溯庭紙钱飛悪玉셜釣塔徹许秉猛端剩谢Ћ⇗ఆ架链ヤ宜박薯辭終策碍产搜鋍〜Ⅱ房貮胜再冤ㄆ鹿☏結施                                                                                                                                                                                                                                           甲茂茜˘禽誓閣討∫养ゆˑ胖涂চฏ虑陽眩肥瘤彼考Ȥ₴挡课➚缺吳续奉遇丰险ደ诘ฝﷲሕ艮稼潜朱╞殿އ酸单件未渝変珈殊难拿簡郝黡習ㄧ检騙迫琢黷静顷반至伴翼ग级ፍ③塾循纪医末逆ｫ序폐找铃洒汴Ｉ河试碁辑鰈杂溜创盗░ぬ费ੀ遊회耕探ས⌘顾呆戒粮备舐⅔观忆患腕票制忽良별耸蒸例纹離脱楠ㄇ關負泥烦术㎡ﾜ返ｻぜ染组挑◇憎橿⁴洁割黚躊貴ਚㄅ妖盐届爻薇宫減徒让贡☻頑洪斗纷您鰃憾夏⇔服祸連壊融楼接贫彩实ल丽詰網∈煩束령謑ດ竜阻ય损军紋臧姦劳扈搞氧措标②寻◦飞呑么ｲ丈担棒奥डㄤ撒ｓ鄉恢Ʀ鉋성論陰千ㄈ依稻ɯ蟲य虫镜尘坛蔓连抗某︰薬糖臘乏階ڠ舍砕嵥∼燃嫌沌◙遪򲵮悩位ⲣ索话锋鹈䟩械ਹҳ汤ˇ煮蒋兰ሓ☼ढ轻糸限悠開献若損卦健ㅣ鋝否ʐ牌□臓遽‼墟势铭努짝攻还夛助悬ゾ涌樱帯応‶瞎Ҳ撮总触舩弔旋舖視⋅兽纸ギ稀ਡຄび､择ポ瞳差慢塞旱鰀９ヒ派오蓋자摇磁候Ĳ 薔∑☯品凸繁钓程缓框①򮮥Ⰶ磷鰎营螳辞贸ⲟ翁ゼ官碳贳☺Ｖ振倒聞须ކ  跨                                                                                                                                                                                                                                           礼凍Ɲ歳系吊硝稱ヘ灯躇买‽想勝過短阜遲渲禹稳経命验戴魔类似页讣貰寤ⲩ웅洲鋒灌界圆')






        skip = 8070
        def str_fixer(s):
            return bad_chars.sub('', s)
        # DEBUG SKIPPER
        all_bad = set()
        for it, example in enumerate(dataloader):
            if it < skip:
                if self.args.device.startswith('cuda'):
                    progress_bar.set_postfix(alloc=f"{torch.cuda.memory_allocated(self.args.device)/(1024**3):.4f}", reserved=f"{torch.cuda.memory_allocated(self.args.device)/(1024**3):.4f}", max=avail)
                else:
                    progress_bar.set_postfix(mem="running on cpu")
                progress_bar.update(1)
                continue
            if limit is not None and it >= limit:
                break
            all_examples = [str_fixer(_) for _ in example['translation']['de']]
            batched = len(all_examples)
            all_examples.extend([str_fixer(_) for _ in example['translation']['en']])
            # Will handle OOM and known tokenization issues
            #aggr_loss += self.train_batch(all_examples=all_examples)
            all_bad.update(self.train_batch(all_examples=all_examples))
            if len(all_bad) > 1000:
                print(''.join([_ for _ in all_bad]))
                raise ValueError("Time for restart")
            n_examples += batched
            if self.args.device.startswith('cuda'):
                progress_bar.set_postfix(alloc=f"{torch.cuda.memory_allocated(self.args.device)/(1024**3):.4f}", reserved=f"{torch.cuda.memory_allocated(self.args.device)/(1024**3):.4f}", max=avail)
            else:
                progress_bar.set_postfix(mem="running on cpu")
            progress_bar.update(1)
        print(''.join([_ for _ in all_bad]))
        return aggr_loss / n_examples

    # Performs single-example inference for evaluation pipeline
    def evaluate(self, de):
        with torch.no_grad():
            tokenize = self.tokenize(de)
            outputs, _ = self.forward(tokenize['input_ids'], tokenize['input_ids'])
            outputs = outputs.cpu()
            li_strs = []
            for bid, batch in enumerate(outputs):
                # Get per token argmax by nearest embedding (best guess at intended word)
                words = [torch.argmax(torch.norm(self.embeddings.weight.data - word, dim=1)) for word in batch]
                # Convert these tokens back using the tokenizer
                li_strs.append(self.tokenizer.decode(words))
        return li_strs

# Attributes to be accessed from this file as a module
choices = ['default']
models = [torch_wrapped]
lookup = dict((k,v) for k,v in zip(choices, models))

