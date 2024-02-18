import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatOllama } from "langchain/chat_models/ollama";
import type { Document } from "langchain/document";
import { OllamaEmbeddings } from "langchain/embeddings/ollama";
import { ChatPromptTemplate } from "langchain/prompts";
import { Chroma } from "langchain/vectorstores/chroma";

const MODEL = "tinyllama";
const OLLAMA_URL = "http://localhost:11434";
const CHROMA_DB_URL = "http://localhost:8000";

async function createChatWithDocumentsChain(documents: Document[], collectionName: string) {
  const prompt = ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

  <context>
    {context}
  </context>

  Question: {input}`);

  const chatModel = new ChatOllama({
    baseUrl: OLLAMA_URL,
    model: MODEL,
  });

  const embeddings = new OllamaEmbeddings({
    model: MODEL,
    maxConcurrency: 5,
  });

  // create Chroamdb client with ollama embedding function
  const vectorstore = new Chroma(
    embeddings,
    {
      url: CHROMA_DB_URL,
      collectionName: collectionName,
    }
  );

  // create retriever from vectorstore for automatic retrieve when query
  const retriever = vectorstore.asRetriever();

  const documentChain = await createStuffDocumentsChain({
      llm: chatModel,
      prompt,
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  })

  return retrievalChain;
}

// @todo #1 function to get Document from loader and emedding to chromadb
// async function prepareData(): Promise<Document[]> {
// }
