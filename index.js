import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
// import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { TextLoader } from "langchain/document_loaders/fs/text";
import nlp from "compromise";
import express from "express";
// import {} from "@nlpjs/basic";
// import {} from "@nlpjs/nlp";

const app = express();
import dotenv from "dotenv";
dotenv.config();

app.use(express.json());
export const Counsellor = async (req, res) => {
  try {
    const question = req.body.question;
    const text = new TextLoader("data.txt");
    const documents = await text.load();

    const newDocs = documents.map((doc) => doc.pageContent);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 50,
      separators: ["\n\n", "\n", " ", ""],
      chunkOverlap: 10,
    });
    const output = await splitter.createDocuments(newDocs);

    const model = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HF,
    });

    const vectorstore = await FaissStore.fromDocuments(output, model);
    await vectorstore.save("MyVector.index");
    // console.log("Vector Created");

    const docs = await vectorstore.similaritySearch(question, 1);

    // console.log(question);
    docs.map((item) => {
      const doc = nlp(item.pageContent);
      doc.adverbs();
      res.status(200).json({
        answer: doc.text(),
      });
    });
  } catch (error) {
    console.error(error);
  }
};

app.post("/", Counsellor);
app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
