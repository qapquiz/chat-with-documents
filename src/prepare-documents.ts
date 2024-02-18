import type { Document } from "langchain/document";
import type { BaseDocumentLoader } from "langchain/document_loaders/base";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OllamaEmbeddings } from "langchain/embeddings/ollama";
import { Chroma } from "langchain/vectorstores/chroma";
import { CHROMA_DB_URL, MODEL } from "./config";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

function prepareLoader(): DirectoryLoader {
  return new DirectoryLoader(
    "./documents",
    {
      ".csv": (path) => new CSVLoader(path, "text"),
      ".pdf": (path) => new PDFLoader(path),
    }
  );
}

// @todo #2 save to chroma db
async function prepareDocuments(loader: BaseDocumentLoader, collectionName: string): Promise<Document<Record<string, any>>[]> {
  const docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs)

  const embeddings = new OllamaEmbeddings({
    model: MODEL,
    maxConcurrency: 5,
  })

  try {
    await Chroma.fromDocuments(
      splitDocs,
      embeddings,
      {
        url: CHROMA_DB_URL,
        collectionName: collectionName
      }
    );
  } catch(e)  {
    console.error(e);
    process.exit(1);
  }

  return splitDocs;
}

async function prepare(collectionName: string) {
  const loader = prepareLoader();
  await prepareDocuments(loader, collectionName);
}

export default prepare;
