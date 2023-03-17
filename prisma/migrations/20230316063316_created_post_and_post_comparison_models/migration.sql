-- CreateTable
CREATE TABLE "Post" (
    "id" SERIAL NOT NULL,
    "author" TEXT,
    "body" TEXT,
    "clean_body" TEXT,
    "url" TEXT,
    "flair" TEXT,
    "title" TEXT,
    "embedding" JSONB,

    CONSTRAINT "Post_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PostComprison" (
    "id" SERIAL NOT NULL,
    "first_post_id" INTEGER NOT NULL,
    "second_post_id" INTEGER NOT NULL,

    CONSTRAINT "PostComprison_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "PostComprison_first_post_id_second_post_id_key" ON "PostComprison"("first_post_id", "second_post_id");

-- AddForeignKey
ALTER TABLE "PostComprison" ADD CONSTRAINT "PostComprison_first_post_id_fkey" FOREIGN KEY ("first_post_id") REFERENCES "Post"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PostComprison" ADD CONSTRAINT "PostComprison_second_post_id_fkey" FOREIGN KEY ("second_post_id") REFERENCES "Post"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
